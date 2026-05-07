"""
DataInsightAgent
------------------------------------
Profiles each table in the store, plus the discovered/confirmed
relationships, and asks the LLM to produce structured observations.
"""
from __future__ import annotations

import io

import pandas as pd

from ..core.agent_base import Agent
from ..core.context import AnalysisContext
from ..core.logging_util import get_logger, timed


SYSTEM_PROMPT = """You are the DATA INSIGHT agent in a multi-agent data analyst.

You are given:
  - A profile of one or more tables in a SQLite store.
  - A list of relationships between tables. Some are *confirmed* (treat as
    real foreign keys); others are *candidates* the user has not approved.

Return a JSON object with:
  - "tables": dict mapping table_name -> {"shape", "key_observations":[...]}
  - "joinable_questions": list of analytical questions that span tables
    (only if there are 2+ tables and at least one confirmed/candidate FK)
  - "single_table_questions": list of questions answerable from one table
  - "relationship_notes": short plain-English description of confirmed and
    high-confidence candidate FKs (e.g. "orders.customer_id appears to
    reference customers.id with 100% containment")

Respond ONLY with JSON. Do NOT invent column names or relationships."""


def _profile_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> str:
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n")
    buf.write("Columns (name, dtype, non_null, n_unique):\n")
    for col in df.columns:
        nn = df[col].notna().sum()
        nu = df[col].nunique(dropna=True)
        buf.write(f"  - {col}: {df[col].dtype}, non_null={nn}, n_unique={nu}\n")
    num = df.select_dtypes(include="number")
    if not num.empty:
        buf.write("\nNumeric describe():\n")
        buf.write(num.describe().round(3).to_string())
        buf.write("\n")
    buf.write(f"\nHead ({sample_rows}):\n")
    buf.write(df.head(sample_rows).to_string())
    return buf.getvalue()


def _relationship_block(ctx: AnalysisContext) -> str:
    lines: list[str] = []
    if ctx.confirmed_relationships:
        lines.append("CONFIRMED relationships (treat as real FKs):")
        for r in ctx.confirmed_relationships:
            lines.append(f"  - {r.label()}  (confidence={r.confidence})")
    candidate_only = [c for c in ctx.relationship_candidates
                      if c not in ctx.confirmed_relationships]
    if candidate_only:
        lines.append("CANDIDATE relationships (NOT confirmed):")
        for r in candidate_only[:10]:
            lines.append(f"  - {r.label()}  (confidence={r.confidence}, "
                         f"kind={r.kind})")
    if not lines:
        lines.append("No relationships discovered.")
    return "\n".join(lines)


class DataInsightAgent(Agent):
    name = "data_insight"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, llm_config=None):
        super().__init__(llm_config)
        self._log = get_logger(self.name)

    def run(self, ctx: AnalysisContext) -> None:
        # A per-table profile section.
        table_profiles = []
        if ctx.store is not None:
            for tm in ctx.store.list_tables():
                head_df = ctx.store.preview(tm.name, n=20)
                profile = _profile_dataframe(head_df)
                table_profiles.append(f"=== TABLE: {tm.name} "
                                      f"(file: {tm.source_filename}) ===\n{profile}")
        elif ctx.df is not None:
            # Single-df fallback
            table_profiles.append(_profile_dataframe(ctx.df))

        with timed(self._log, "build profile"):
            full_profile = "\n\n".join(table_profiles)

        rel_block = _relationship_block(ctx) if ctx.store is not None else ""

        user_msg = (
            f"User's question: {ctx.user_query}\n\n"
            f"TABLES:\n{full_profile}\n\n"
            f"RELATIONSHIPS:\n{rel_block}"
        )
        with timed(self._log, "summarise via LLM"):
            raw = self.client.complete(self.system_prompt, user_msg)
        ctx.df_summary = self.extract_json(raw)
        self._log.info(
            f"summary keys: {list(ctx.df_summary.keys())}",
            extra={"payload": str(ctx.df_summary)},
        )
        ctx.log(self.name, f"Summarised {len(table_profiles)} table(s)")