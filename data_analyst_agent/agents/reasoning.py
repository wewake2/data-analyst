"""
ReasoningAgent
--------------
Takes the original question, the executed result, and any errors, and
produces:

1. `ctx.explanation` : a short natural-language answer.
2. `ctx.claims` : a structured list of {text, evidence_kind, evidence_ref}
   where `evidence_ref` points at a concrete cell or scalar in
   `ctx.analysis_result`. The provenance verifier resolves each ref
   against the actual result and flags any that don't match.

The structured output is for the inline citations in the UI.
The freeform prose is the fallback if structured output fails to parse
or if a claim's evidence can't be resolved.
"""
from __future__ import annotations

import json
import pandas as pd

from ..core.agent_base import Agent
from ..core.context import AnalysisContext
from ..core.logging_util import get_logger, timed

SYSTEM_PROMPT = """You are the REASONING agent in a multi-agent data analyst.

Given a user question and the output of code that was just executed, produce
a JSON object with two fields:

{
  "summary": "<one or two sentence high-level answer>",
  "claims": [
    {
      "text": "<one factual sentence>",
      "measurements": [<list of numbers in `text` that are factual claims>],
      "evidence_kind": "cell" | "scalar" | "row" | "column" | "shape" | "stat",
      "evidence_ref": <see below>
    },
    ...
  ]
}

EVIDENCE REFERENCE FORMATS by evidence_kind:
  - "cell":   {"row": <row label or index int>, "col": "<column name>"}
              For: "North region revenue was 87432" -> row="North", col="revenue"
  - "scalar": {"value": <number or string>}
              For: result is a single scalar (e.g. 'mean = 12.4')
  - "row":    {"row": <row label or index int>}
              For: claims about an entire row
  - "column": {"col": "<column name>"}
              For: claims about an entire column
  - "shape":  {}
              For: claims about result.shape - total rows or total columns
  - "stat":   {"stat": "min"|"max"|"mean"|"median"|"sum"|"count"|"std"|"var",
               "col": "<column name>"}
              For: claims like "the maximum revenue was X"

CHOOSING evidence_kind:

1. The evidence_kind is determined by what NUMBER you are asserting, not
   by the subject of the claim.
   - If the number is a row/column count -> "shape"
   - If the number is a min/max/mean/median/sum/std/var/count of a column -> "stat"
   - If the number is a single value at a specific (row, col) -> "cell"

2. PREFER "stat" over "cell" for superlatives ("highest", "lowest", "most",
   "least", "max", "min", "average", "total"). Use "cell" only when the
   claim names a specific row label AND a specific column.

3. SPLIT compound claims. A claim like "the top 20 products have rating 5"
   makes TWO assertions: (a) there are 20 rows, (b) every row has rating=5.
   Emit TWO separate claims - one with kind="shape" for the count, one with
   kind="column" or "stat" for the value.

MEASUREMENTS:
List ONLY the numbers in `text` that assert a fact about the data. Numbers
that are labels, positions, or context go in the text but NOT in
measurements:
    - Calendar dates: "January 2025" -> 2025 is a label
    - Percentile labels: "75% of products are at or below $210" -> 75 is a label
    - Ordinals/positions: "the top 3 categories" -> 3 is a label (it's naming a slice, not asserting a fact)
    - Counts of result structure: "across 11 months" -> 11 is a label IF you've already made a separate kind="shape" claim asserting it; if not, include it as a measurement on the shape claim.
    - Filter thresholds: "products with rating > 4" -> 4 is a threshold, not a measurement. The measurement is whatever count or value the filter PRODUCED.
Only include numbers that, if wrong, would make the claim factually false.

USING THE VERIFIED FACTS BLOCK:
The user message includes a "Verified facts" section, computed
deterministically from the FULL result. The "Result value" preview is
TRUNCATED - typically only the first few rows.

When making numeric claims:
- Row counts MUST match the row count in Verified facts.
- Min/max/mean/median/sum claims MUST use the values in Verified facts.
- Trust Verified facts over the preview.
- For values not in Verified facts (e.g. specific cells in long results), cite only what is visible in the preview.
- If you cannot see a value AND it is not in Verified facts, omit the claim. Vague-but-true beats precise-but-wrong.

CRITICAL - build evidence_refs against the RESULT structure, not the input
tables. If the user asked about "prices" but the result is a `.describe()`
output with rows like 'min', 'max', 'mean', then evidence_refs must
reference the RESULT's column names and row labels - not the original
input column 'price'.

RULES:
- Every value in `measurements` MUST appear in the actual result.
  Do not invent or round to fictional numbers.
- 3-6 claims per response. Each claim must have evidence.
- Use exact column names and row labels from the result; do not paraphrase.
- If the result is empty/None, emit measurements: [] for an emptiness claim.
- If there was an execution error, emit one claim with evidence_kind="scalar"
  and evidence_ref={"value": "error"}, and explain in summary.
- Do not make claims about the INPUT table's row count. Your claims must be verifiable against the RESULT only. If the user asks about
  "distribution of prices" and the result is binned, claims should be about the bins, not about the input table.
- Do NOT make claims that require arithmetic on result values. If you want to say "X products have no reviews," verify it directly via a
  COUNT of NULL ratings, not by subtraction.

EXAMPLES:

  text: "March had the highest signups with 481 users in 2025"
  measurements: [481]              # 2025 is a date label, not a measurement
  evidence_kind: "stat"
  evidence_ref: {"stat": "max", "col": "user_count"}

  text: "There are 6,872 cities with order data"
  measurements: [6872]
  evidence_kind: "shape"
  evidence_ref: {}

  text: "Electronics had the highest revenue at $4,961,737"
  measurements: [4961737]
  evidence_kind: "stat"
  evidence_ref: {"stat": "max", "col": "total_revenue"}

Respond ONLY with the JSON object. No prose outside the JSON."""

def _stringify(result, max_chars: int = 4000) -> str:
    if result is None:
        return "<no result>"
    try:
        s = result.to_string() if hasattr(result, "to_string") else repr(result)
    except Exception:
        s = repr(result)
    return s if len(s) <= max_chars else s[:max_chars] + "\n... (truncated)"


def _result_facts(result) -> str:
    """
    Deterministic facts about the result. The LLM should cite these
    rather than computing them from a truncated preview.
    """
    if result is None:
        return "(no result)"

    lines = []
    if isinstance(result, pd.DataFrame):
        lines.append(f"FACT: result has exactly {len(result)} rows and "
                     f"{len(result.columns)} columns.")
        for col in result.select_dtypes(include="number").columns:
            s = result[col].dropna()
            if len(s) == 0:
                continue
            lines.append(
                f"FACT: column '{col}' - "
                f"min={s.min()}, max={s.max()}, "
                f"mean={s.mean():.4f}, median={s.median()}, "
                f"sum={s.sum()}, count={len(s)}"
            )
    elif isinstance(result, pd.Series):
        lines.append(f"FACT: result is a Series with exactly {len(result)} entries.")
        if pd.api.types.is_numeric_dtype(result):
            s = result.dropna()
            if len(s) > 0:
                lines.append(
                    f"FACT: values - min={s.min()}, max={s.max()}, "
                    f"mean={s.mean():.4f}, median={s.median()}, "
                    f"sum={s.sum()}, count={len(s)}"
                )
    return "\n".join(lines) if lines else "(no facts derivable)"


def _result_schema(result) -> str:
    """
    Describe the SHAPE of result so the LLM picks valid evidence_refs.

    For DataFrame: list columns and a few index labels.
    For Series:   say 'Series' and list index labels (these are the row keys).
    For scalar:   say it's a scalar.
    """
    if result is None:
        return "result is None (likely an error upstream)"
    if isinstance(result, pd.DataFrame):
        idx_sample = list(result.index[:10])
        cols = list(result.columns)
        return (f"result is a DataFrame with shape {result.shape}.\n"
                f"  columns:       {cols}\n"
                f"  index labels (first 10): {idx_sample}\n"
                f"  -> for cell refs, use one of these column names exactly\n"
                f"  -> for stat refs (min/max/etc.), 'col' must be one of: {cols}")
    if isinstance(result, pd.Series):
        idx_sample = list(result.index[:15])
        return (f"result is a Series of length {len(result)}.\n"
                f"  the Series has NO named columns. Each element is a value\n"
                f"  reachable by its index label.\n"
                f"  index labels (first 15): {idx_sample}\n"
                f"  -> for cell refs, set col=null or omit it; row=<index label>\n"
                f"  -> stat refs (min/max/mean) on a Series do NOT need 'col';"
                f"     omit it or set col=null")
    if isinstance(result, dict):
        return (f"result is a dict with keys {list(result.keys())[:10]}.\n"
                f"  -> for cell refs, use 'row' = an inner key; 'col' = an outer key")
    return f"result is a scalar of type {type(result).__name__} = {result!r}"

class ReasoningAgent(Agent):
    name = "reasoning"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, llm_config=None):
        super().__init__(llm_config)
        self._log = get_logger(self.name)

    def run(self, ctx: AnalysisContext) -> None:
        parts = [
            f"User question: {ctx.user_query}",
            f"Query plan: {ctx.query_understanding}",
            f"Code that ran:\n{ctx.analysis_code}",
        ]
        if ctx.analysis_error:
            parts.append(f"Result schema:\n{_result_schema(ctx.analysis_result)}")
            parts.append(f"ERROR during execution:\n{ctx.analysis_error}")
        else:
            parts.append(f"Result schema:\n{_result_schema(ctx.analysis_result)}")
            parts.append(f"Verified facts (cite these - do NOT recompute):\n"
                         f"{_result_facts(ctx.analysis_result)}")
            parts.append(f"Result value (preview, may be truncated):\n"
                         f"{_stringify(ctx.analysis_result)}")
        if ctx.plot_error:
            parts.append(f"Plot error:\n{ctx.plot_error}")
        elif ctx.plot_path:
            parts.append(f"A chart was rendered at: {ctx.plot_path}")

        user_msg = "\n\n".join(parts)
        with timed(self._log, "reason via LLM"):
            raw = self.client.complete(self.system_prompt, user_msg).strip()

        # fall back to treating the whole thing as
        # a single un-cited claim if the LLM didn't follow the schema.
        parsed = self.extract_json(raw)
        summary = parsed.get("summary")
        claims = parsed.get("claims")

        if not isinstance(claims, list) or not summary:
            self._log.warning(
                "structured reasoning failed; falling back to prose-only",
                extra={"payload": raw},
            )
            ctx.explanation = raw
            ctx.claims = []
        else:
            ctx.explanation = summary.strip()
            ctx.claims = claims
            self._log.info(
                f"got {len(claims)} structured claims",
                extra={"payload": json.dumps(claims, indent=2, default=str)},
            )

        ctx.log(self.name, f"Generated explanation with {len(ctx.claims)} claims")


