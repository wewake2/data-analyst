"""
AnalysisContext is the shared state passed between agents.

Multi-table version:
  - `tables` is the canonical source of truth (TableStore + per-table metas)
  - `df` is kept as a convenience for the single-table case (= the only
    table's DataFrame), so single-CSV demos still work without changes.

Each agent reads what it needs and writes its output. The orchestrator
owns the lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .table_store import TableStore


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AnalysisContext:
    # Inputs
    user_query: str
    # Multi-table state
    store: Optional["TableStore"] = None
    confirmed_relationships: list = field(default_factory=list)
    relationship_candidates: list = field(default_factory=list)

    # Single-table convenience: the lone DataFrame for n=1 cases.
    # When n > 1, this stays None and code generation uses the store instead.
    df: Optional[pd.DataFrame] = None

    # Filled by DataInsightAgent
    df_summary: Optional[dict] = None

    # Filled by CodeGenerationAgent
    query_understanding: Optional[dict] = None
    analysis_code: Optional[str] = None
    plot_code: Optional[str] = None
    engine: str = "pandas"   # "pandas" | "sql" - chosen by query understanding

    # Filled by ExecutionAgent
    analysis_result: Any = None
    analysis_stdout: str = ""
    analysis_error: Optional[str] = None
    plot_path: Optional[str] = None
    plot_error: Optional[str] = None

    # Filled by ReasoningAgent
    explanation: Optional[str] = None
    claims: list = field(default_factory=list)
    resolved_claims: list = field(default_factory=list)

    # Filled by run_auto_checks at end of pipeline
    checks: list = field(default_factory=list)

    # Token usage tracking
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    # Free-form trace for debugging / UI
    trace: list = field(default_factory=list)

    # ---- helpers -----------------------------------------------------------

    @property
    def is_multi_table(self) -> bool:
        return self.store is not None and len(self.store.list_tables()) > 1

    @property
    def all_tables(self) -> list:
        if self.store is None:
            return []
        return self.store.list_tables()

    def log(self, agent: str, message: str) -> None:
        self.trace.append({"agent": agent, "message": message})