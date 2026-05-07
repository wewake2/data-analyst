"""
LLM client abstraction.
Each agent can have its own LLM config (provider + model + params),
or fall back to the default registered on the orchestrator.
Supports: Anthropic Claude, OpenAI, and a "MockLLM" for offline testing
of the wiring without burning API credits.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class LLMConfig:
    """Per-agent LLM configuration. Any field left None falls back to default."""
    provider: Optional[str] = None  # "anthropic" | "openai"
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def merged_with(self, default: "LLMConfig") -> "LLMConfig":
        """Return a config where None fields are filled from `default`."""
        return LLMConfig(
            provider=self.provider or default.provider,
            model=self.model or default.model,
            temperature=self.temperature if self.temperature is not None else default.temperature,
            max_tokens=self.max_tokens or default.max_tokens,
            api_key=self.api_key or default.api_key,
            extra={**default.extra, **self.extra},
        )


@dataclass
class TokenCount:
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    """Minimal interface every concrete LLM backend must satisfy."""
    def complete(self, system: str, user: str, **kwargs: Any) -> str: ...


class AnthropicClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_token_count: TokenCount = TokenCount()
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from e
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        resp = self._client.messages.create(
            model=self.config.model or "claude-sonnet-4-5",
            max_tokens=self.config.max_tokens or 2048,
            temperature=self.config.temperature if self.config.temperature is not None else 0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.last_token_count = TokenCount(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


class OpenAIClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_token_count: TokenCount = TokenCount()
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from e
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model or "gpt-4o-mini",
            temperature=self.config.temperature if self.config.temperature is not None else 0.2,
            max_tokens=self.config.max_tokens or 2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if resp.usage:
            self.last_token_count = TokenCount(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        return resp.choices[0].message.content or ""


class BedrockClient:
    """
    AWS Bedrock client supporting both Anthropic Claude and NVIDIA Nemotron.

    Routes by model id prefix:
      - 'anthropic.*' / 'us.anthropic.*' / 'global.anthropic.*'
            -> Bedrock Runtime Converse API (handles temperature/top_p quirks)
      - 'nvidia.nemotron-*'
            -> Bedrock OpenAI-compatible Mantle endpoint
              (https://bedrock-mantle.<region>.api.aws/v1)

    Region is taken from config.extra['aws_region'] or AWS_REGION env var.
    Credentials come from the standard boto3/AWS chain (env vars, ~/.aws,
    instance profile, etc.) - we don't pass api_key for Bedrock.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_token_count: TokenCount = TokenCount()
        self.model_id = config.model or "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        self.region = config.extra.get("aws_region") or os.getenv("AWS_REGION", "us-east-1")
        self._kind = self._classify(self.model_id)

        if self._kind == "claude":
            try:
                import boto3  # type: ignore
            except ImportError as e:
                raise ImportError("boto3 required: pip install boto3") from e
            self._bedrock = boto3.client("bedrock-runtime", region_name=self.region)
        elif self._kind == "nemotron":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:
                raise ImportError("openai required for Nemotron via Mantle: pip install openai") from e
            api_key = config.api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            if not api_key:
                raise RuntimeError(
                    "Nemotron via Bedrock Mantle needs a Bedrock bearer token. "
                    "Set AWS_BEARER_TOKEN_BEDROCK or pass api_key in LLMConfig."
                )
            self._oa = OpenAI(
                api_key=api_key,
                base_url=f"https://bedrock-mantle.{self.region}.api.aws/v1",
            )

    @staticmethod
    def _classify(model_id: str) -> str:
        m = model_id.lower()
        # Strip cross-region prefixes (us./eu./apac./global./jp.)
        for prefix in ("global.", "us.", "eu.", "apac.", "jp."):
            if m.startswith(prefix):
                m = m[len(prefix):]
                break
        if m.startswith("anthropic."):
            return "claude"
        if m.startswith("nvidia.nemotron"):
            return "nemotron"
        raise ValueError(f"Unsupported Bedrock model id: {model_id}")

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        if self._kind == "claude":
            return self._complete_claude(system, user)
        return self._complete_nemotron(system, user)

    def _complete_claude(self, system: str, user: str) -> str:
        # Sonnet 4.5 quirk: only ONE of temperature / top_p, not both.
        inference_config: dict = {"maxTokens": self.config.max_tokens or 2048}
        if self.config.temperature is not None:
            inference_config["temperature"] = self.config.temperature
        else:
            inference_config["temperature"] = 0.2

        resp = self._bedrock.converse(
            modelId=self.model_id,
            system=[{"text": system}] if system else [],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig=inference_config,
        )
        usage = resp.get("usage", {})
        self.last_token_count = TokenCount(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )
        out_blocks = resp["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in out_blocks)

    def _complete_nemotron(self, system: str, user: str) -> str:
        # Nemotron Super is a thinking model - for code-gen we want
        # reasoning OFF by default to keep output deterministic and short.
        # Caller can override via config.extra['enable_thinking'].
        enable_thinking = self.config.extra.get("enable_thinking", False)
        resp = self._oa.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature if self.config.temperature is not None else 0.2,
            max_tokens=self.config.max_tokens or 2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        if resp.usage:
            self.last_token_count = TokenCount(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        return resp.choices[0].message.content or ""


class MockLLM:
    """
    Deterministic offline LLM for testing the agent wiring.

    Returns hand-crafted responses based on which agent is calling
    (detected via markers in the system prompt). Lets you run the full
    pipeline end-to-end without any network or API key.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_token_count: TokenCount = TokenCount()

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        s = system.lower()
        # Match on the explicit agent role markers in the system prompts.
        # Order: most specific first; "REASONING agent" must beat "summary".
        if "reasoning agent" in s:
            return json.dumps({
                "summary": "The analysis described the dataset across the "
                           "requested dimensions. See the result for the "
                           "full breakdown.",
                "claims": [
                    {
                        "text": "The result has multiple columns describing "
                                "the dataset.",
                        "evidence_kind": "shape",
                        "evidence_ref": {},
                    },
                ],
            })
        if "data insight" in s:
            return json.dumps({
                "shape": "Inferred from prompt",
                "key_observations": [
                    "Numeric columns present - useful for aggregation",
                    "At least one categorical column for grouping",
                ],
                "potential_questions": [
                    "What is the distribution of the numeric columns?",
                    "How do values vary across categories?",
                ],
            })
        if "query understanding" in s:
            return json.dumps({
                "intent": "aggregate_and_visualize",
                "target_columns": ["<inferred from question>"],
                "operation": "groupby + aggregate",
                "needs_plot": True,
            })
        if "plot code" in s:
            return (
                "```python\n"
                "import matplotlib.pyplot as plt\n"
                "fig, ax = plt.subplots()\n"
                "df.select_dtypes(include='number').sum().plot(kind='bar', ax=ax)\n"
                "ax.set_title('Sum by numeric column')\n"
                "```"
            )
        if "code writing" in s:
            return (
                "```python\n"
                "import pandas as pd\n"
                "result = df.describe(include='all')\n"
                "```"
            )
        return "OK"


def build_client(config: LLMConfig) -> LLMClient:
    """Factory: pick the right backend based on the merged config."""
    provider = (config.provider or "mock").lower()
    if provider == "anthropic":
        inner = AnthropicClient(config)
    elif provider == "openai":
        inner = OpenAIClient(config)
    elif provider == "bedrock":
        inner = BedrockClient(config)
    elif provider == "mock":
        inner = MockLLM(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return LoggingClient(inner, config)


class LoggingClient:
    """
    Wraps any LLMClient and logs every call (provider/model, prompt sizes,
    response size, latency, token usage). The wrapped client is fully transparent.
    """
    def __init__(self, inner: "LLMClient", config: LLMConfig):
        self._inner = inner
        self._config = config
        self.last_token_count: TokenCount = TokenCount()
        self.total_token_count: TokenCount = TokenCount()
        from .logging_util import get_logger
        self._log = get_logger("llm")

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        import time as _time
        provider = self._config.provider or "?"
        model = self._config.model or "?"
        self._log.info(
            f"-> {provider}/{model}  sys={len(system)}c user={len(user)}c",
            extra={"payload": f"SYSTEM:\n{system}\n\nUSER:\n{user}"},
        )
        t0 = _time.perf_counter()
        try:
            out = self._inner.complete(system, user, **kwargs)
            dt_ms = (_time.perf_counter() - t0) * 1000
            self.last_token_count = getattr(self._inner, "last_token_count", TokenCount())
            self.total_token_count.input_tokens += self.last_token_count.input_tokens
            self.total_token_count.output_tokens += self.last_token_count.output_tokens
            tok = self.last_token_count
            self._log.info(
                f"<- {provider}/{model}  {len(out)}c in {dt_ms:.0f} ms  "
                f"tokens: {tok.input_tokens} in / {tok.output_tokens} out",
                extra={"payload": out, "duration_ms": dt_ms},
            )
            return out
        except Exception as e:
            dt_ms = (_time.perf_counter() - t0) * 1000
            self._log.error(
                f"!! {provider}/{model} failed in {dt_ms:.0f} ms: {e}",
                extra={"duration_ms": dt_ms},
            )
            raise