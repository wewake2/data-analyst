"""
Base class for all agents.

Each agent owns its name, system prompt, and an *optional* LLMConfig.
At construction time inside the orchestrator the config is merged with
the orchestrator's default - so an unset agent inherits the default,
and a partially-set agent overrides only the fields it specifies.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from .context import AnalysisContext
from .llm import LLMClient, LLMConfig


class Agent(ABC):
    name: str = "agent"
    system_prompt: str = ""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        # Per-agent override; orchestrator will merge with default and build the actual client via `bind_client`.
        self.llm_config = llm_config or LLMConfig()
        self._client: Optional[LLMClient] = None

    def bind_client(self, client: LLMClient) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            raise RuntimeError(f"Agent '{self.name}' has no LLM client bound. "
                               f"Use Orchestrator to construct agents.")
        return self._client

    @abstractmethod
    def run(self, ctx: AnalysisContext) -> None: ...

    # --- helpers shared across agents ---

    @staticmethod
    def extract_code(text: str, language: str = "python") -> str:
        """
        Pull a fenced code block out of an LLM response.
        Falls back to the whole text if no fence is present.
        """
        pattern = rf"```(?:{language})?\s*\n?(.*?)```"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return (m.group(1) if m else text).strip()

    @staticmethod
    def extract_json(text: str) -> dict:
        """
        Parse JSON out of an LLM response, tolerating ```json fences and
        leading/trailing prose.
        """
        # Try fenced first
        m = re.search(r"```(?:json)?\s*\n?(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        candidate = m.group(1) if m else text.strip()
        # Find the outermost {...} if there's still surrounding prose
        if not candidate.startswith("{") and not candidate.startswith("["):
            obj = re.search(r"\{.*\}", candidate, re.DOTALL)
            if obj:
                candidate = obj.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {"_raw": text}