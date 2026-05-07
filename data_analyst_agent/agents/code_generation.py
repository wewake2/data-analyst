"""
CodeGenerationAgent (multi-table aware)
---------------------------------------
Sub-stages :
  1. Query understanding   -> structured plan including:
       - engine : "sql" | "pandas"
       - tables_needed : list[str]
       - joins : list[{left, right, on, how, rationale}]
       - needs_plot, plot_type
  2. Analysis code - SQL string OR pandas code
  3. Plot code - matplotlib (always pandas)

Engine choices :
  - SQL is concise for joins, groupBys, multi-table aggregates
  - pandas is better for : rolling windows, complex apply, anything stateful
  - For single-table contexts the LLM still picks; pandas usually wins
"""
from __future__ import annotations

from ..core.agent_base import Agent
from ..core.context import AnalysisContext
from ..core.logging_util import get_logger, timed


QUERY_UNDERSTANDING_PROMPT = """You are the QUERY UNDERSTANDING sub-agent.

You will be given the user's question and a schema describing one or more
tables (with optional confirmed foreign keys).

Translate the question into a structured plan. Return JSON with:
- "intent": short label (e.g. "describe", "groupby_aggregate", "filter", "join_aggregate", "time_series", "distribution", "compare_groups")
- "engine": "sql" or "pandas"
    * Prefer "sql" when joins or multi-table aggregations are needed.
    * Prefer "pandas" for single-table work, rolling windows, melt/pivot.
- "tables_needed": list of table names from the schema
- "joins": list of objects, each:
    {"left": "<table>", "right": "<table>",
     "on": "<column>" | {"left": "<col>", "right": "<col>"},
     "how": "inner" | "left" | "right" | "outer",
     "rationale": "<one short sentence>"}
  Use only confirmed FKs or column-name matches present in the schema.
- "operation": short description of the aggregation/filter
- "needs_plot": boolean
- "plot_type": one of "histogram","bar","line","scatter","box","heatmap",null

Use exact table and column names from the schema. Respond ONLY with JSON."""


SQL_CODE_PROMPT = """You are the SQL WRITING sub-agent.

Write a single SQL query that answers the user's question against the
provided schema. The query runs against SQLite.

STRICT RULES:
- A single SELECT statement, no CTEs that DDL anything, no multi-statement.
- Reference only tables and columns that appear in the schema.
- Quote identifiers with double quotes if they contain unusual characters.
- Use the joins specified by the query plan, with the same `how` semantics.
- The SELECT must produce columns the user can read directly. No SELECT *.

Respond with a single ```sql fenced block and nothing else."""

CODE_WRITING_PROMPT = """You are the CODE WRITING sub-agent (pandas).

You have access to a dict named `dfs` mapping table name -> pandas DataFrame.
ALWAYS use `dfs["<table_name>"]`. Do NOT use `df` - it may not exist.

STRICT RULES:
- The code MUST assign the final answer to a variable named `result`.
- `result` MUST be a pandas DataFrame, pandas Series, or scalar (int/float/str). Do NOT use `.to_dict()`. Do NOT wrap output in a dict
  or list.
- Use only pandas, numpy, and the Python stdlib. No file/network I/O.
- No imports of matplotlib/seaborn here - plotting is a separate stage.
- No print statements; the runner captures `result`.
- For multi-table queries: use `pd.merge(...)` matching the planned joins.
- Use exact table names from the schema (case-sensitive).
- Use exact column names from the schema (case-sensitive).
- Before .nlargest/.nsmallest/.sort_values on a column, ensure it's numeric: `df['col'] = pd.to_numeric(df['col'], errors='coerce')` if needed.
- After a left/outer merge, aggregate columns may be NaN - handle with .fillna(0) or .dropna() as appropriate.
- After filtering with df[boolean_mask], if you'll modify the result,
  add .copy() to avoid SettingWithCopyWarning.

EXAMPLE for a single-table query:
    result = dfs["products"].groupby("category").size().rename("n")

EXAMPLE for a multi-table query:
    merged = dfs["orders"].merge(dfs["users"], on="user_id", how="inner")
    result = merged.groupby("gender")["total_amount"].mean()

Respond with a single ```python fenced block and nothing else."""

PLOT_CODE_PROMPT = """You are the PLOT CODE sub-agent.

Write Python code that produces ONE matplotlib figure visualizing the
answer. The previously computed answer is `result` (DataFrame/Series/scalar).
Raw tables are available as `dfs` (dict). Do NOT use `df`.

STRICT RULES:
- Use matplotlib only (pyplot is fine; seaborn allowed if installed).
- Create exactly one figure via `fig, ax = plt.subplots(...)`.
- Set a clear title and axis labels.
- Do NOT call plt.show() and do NOT savefig() - the runner handles saving.
- The final figure object MUST be assigned to a variable named `fig`.
- For most plots, just plot `result` directly - it's already aggregated.

Respond with a single ```python fenced block and nothing else."""

def _schema_block(ctx: AnalysisContext) -> str:
    if ctx.store is not None:
        return ctx.store.schema_for_prompt()
    if ctx.df is not None:
        cols = "\n".join(f'  - "{c}"  {ctx.df[c].dtype}' for c in ctx.df.columns)
        return f'TABLE "data" ({len(ctx.df)} rows)\n{cols}'
    return "(no tables loaded)"


class CodeGenerationAgent(Agent):
    name = "code_generation"
    system_prompt = CODE_WRITING_PROMPT

    def __init__(self, llm_config=None):
        super().__init__(llm_config)
        self._log = get_logger(self.name)

    def _understand(self, ctx: AnalysisContext) -> None:
        user_msg = (
            f"User question: {ctx.user_query}\n\n"
            f"DataFrame summary: {ctx.df_summary}\n\n"
            f"Schema:\n{_schema_block(ctx)}"
        )
        with timed(self._log, "understand query"):
            raw = self.client.complete(QUERY_UNDERSTANDING_PROMPT, user_msg)
        ctx.query_understanding = self.extract_json(raw)
        ctx.engine = ctx.query_understanding.get("engine", "pandas").lower()
        if ctx.engine not in ("sql", "pandas"):
            ctx.engine = "pandas"
        self._log.info(
            f"intent={ctx.query_understanding.get('intent')} "
            f"engine={ctx.engine} "
            f"tables={ctx.query_understanding.get('tables_needed')} "
            f"needs_plot={ctx.query_understanding.get('needs_plot')}",
            extra={"payload": str(ctx.query_understanding)},
        )
        ctx.log(self.name, f"Intent: {ctx.query_understanding.get('intent')} "
                           f"(engine: {ctx.engine})")

    def _write_analysis_code(self, ctx: AnalysisContext) -> None:
        if ctx.engine == "sql":
            user_msg = (
                f"User question: {ctx.user_query}\n\n"
                f"Query plan: {ctx.query_understanding}\n\n"
                f"Schema:\n{_schema_block(ctx)}"
            )
            with timed(self._log, "write SQL"):
                raw = self.client.complete(SQL_CODE_PROMPT, user_msg)
            ctx.analysis_code = self.extract_code(raw, language="sql")
        else:
            user_msg = (
                f"User question: {ctx.user_query}\n\n"
                f"Query plan: {ctx.query_understanding}\n\n"
                f"Schema:\n{_schema_block(ctx)}"
            )
            with timed(self._log, "write pandas"):
                raw = self.client.complete(CODE_WRITING_PROMPT, user_msg)
            ctx.analysis_code = self.extract_code(raw, language="python")
        self._log.info(
            f"analysis code ({ctx.engine}): {len(ctx.analysis_code)} chars",
            extra={"payload": ctx.analysis_code},
        )
        ctx.log(self.name, f"Analysis code ({ctx.engine}): "
                           f"{len(ctx.analysis_code)} chars")

    def _write_plot_code(self, ctx: AnalysisContext) -> None:
        plan = ctx.query_understanding or {}
        if not plan.get("needs_plot", False):
            self._log.info("plot not required by query plan")
            ctx.log(self.name, "Plot not required by query plan")
            return
        user_msg = (
            f"User question: {ctx.user_query}\n\n"
            f"Plot type hint: {plan.get('plot_type')}\n\n"
            f"Engine used: {ctx.engine}\n"
            f"Analysis code that produced `result`:\n{ctx.analysis_code}"
        )
        with timed(self._log, "write plot code"):
            raw = self.client.complete(PLOT_CODE_PROMPT, user_msg)
        ctx.plot_code = self.extract_code(raw, language="python")
        self._log.info(
            f"plot code: {len(ctx.plot_code)} chars",
            extra={"payload": ctx.plot_code},
        )
        ctx.log(self.name, f"Plot code: {len(ctx.plot_code)} chars")

    def run(self, ctx: AnalysisContext) -> None:
        self._understand(ctx)
        self._write_analysis_code(ctx)
        self._write_plot_code(ctx)