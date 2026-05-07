"""
Orchestrator - wires LLMs to agents and runs the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..agents.code_generation import CodeGenerationAgent
from ..agents.data_insight import DataInsightAgent
from ..agents.execution import ExecutionAgent
from ..agents.reasoning import ReasoningAgent
from ..core.agent_base import Agent
from ..core.context import AnalysisContext
from ..core.llm import LLMConfig, build_client
from ..core.logging_util import get_logger
from ..core.provenance import resolve_claims
from ..core.relationships import RelationshipCandidate, discover_relationships
from ..core.table_store import TableStore
from ..core.verification import run_auto_checks


@dataclass
class Orchestrator:
    default_llm: LLMConfig = field(default_factory=lambda: LLMConfig(provider="mock"))
    agent_llms: dict = field(default_factory=dict)
    output_dir: str = "/tmp/data_analyst_outputs"

    def __post_init__(self):
        self.agents: dict[str, Agent] = {
            "data_insight":    DataInsightAgent(self.agent_llms.get("data_insight")),
            "code_generation": CodeGenerationAgent(self.agent_llms.get("code_generation")),
            "execution":       ExecutionAgent(output_dir=self.output_dir,
                                              llm_config=self.agent_llms.get("execution")),
            "reasoning":       ReasoningAgent(self.agent_llms.get("reasoning")),
        }
        for name, agent in self.agents.items():
            merged = agent.llm_config.merged_with(self.default_llm)
            agent.bind_client(build_client(merged))

    # ----- runners ---------------------------------------
    def _run_pipeline(self, ctx: AnalysisContext) -> AnalysisContext:
        log = get_logger("orchestrator")
        for name in ("data_insight", "code_generation", "execution", "reasoning"):
            log.info(f"--> {name}")
            self.agents[name].run(ctx)
            client = getattr(self.agents[name], "_client", None)
            if client is not None:
                tok = getattr(client, "total_token_count", None)
                if tok:
                    ctx.token_usage.add(tok.input_tokens, tok.output_tokens)
                    tok.input_tokens = 0
                    tok.output_tokens = 0

        if ctx.claims:
            ctx.resolved_claims = resolve_claims(ctx.analysis_result, ctx.claims)
            n_pass = sum(1 for c in ctx.resolved_claims if c.status == "pass")
            n_warn = sum(1 for c in ctx.resolved_claims if c.status == "warn")
            n_fail = sum(1 for c in ctx.resolved_claims if c.status == "fail")
            log.info(f"provenance: {n_pass} pass / {n_warn} warn / {n_fail} fail")

        ctx.checks = run_auto_checks(ctx)
        for c in ctx.checks:
            log.info(f"check [{c.status:4s}] {c.name}: {c.message}")
        log.info("done")
        return ctx

    def analyze(self, df: pd.DataFrame, query: str) -> AnalysisContext:
        """Legacy single-DataFrame entry point."""
        log = get_logger("orchestrator")
        log.info(f"start analyze: {len(df)} rows x {len(df.columns)} cols  "
                 f"query={query!r}")
        ctx = AnalysisContext(user_query=query, df=df)
        return self._run_pipeline(ctx)

    def analyze_store(
        self,
        store: TableStore,
        query: str,
        confirmed_relationships: Optional[list[RelationshipCandidate]] = None,
    ) -> AnalysisContext:
        """Multi-table entry point with a shared TableStore."""
        log = get_logger("orchestrator")
        tables = store.list_tables()
        log.info(f"start analyze_store: {len(tables)} table(s)  query={query!r}")

        candidates = discover_relationships(store) if len(tables) > 1 else []
        log.info(f"discovered {len(candidates)} candidate relationships")

        ctx = AnalysisContext(
            user_query=query,
            store=store,
            confirmed_relationships=confirmed_relationships or [],
            relationship_candidates=candidates,
        )
        if len(tables) == 1:
            only = tables[0]
            ctx.df = store.query_pandas(f'SELECT * FROM "{only.name}"')
        return self._run_pipeline(ctx)

    # ----- introspection -----------------------
    def describe_setup(self) -> str:
        lines = ["Agent -> LLM mapping:"]
        for name, agent in self.agents.items():
            merged = agent.llm_config.merged_with(self.default_llm)
            lines.append(f"  {name:18s} {merged.provider}/{merged.model}")
        return "\n".join(lines)