"""
Result verification.

Two surfaces:

1. `run_auto_checks(ctx)` - fast, deterministic sanity checks that run server-side after the analysis. Each check returns a Pass/Warn/Fail
   verdict with a short message. Designed to *catch obvious wrongness*, not to formally prove correctness.

2. On-demand verification actions (`compute_describe_baseline`,
   `recompute_with_sample`, `code_in_plain_english_prompt`) that the UI can wire to buttons.

The auto-checks intentionally don't depend on the LLM - they only look at
the dataframe, the code string, and the result object. That keeps them cheap, deterministic, and trustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .context import AnalysisContext


@dataclass
class CheckResult:
    name: str
    status: str          # "pass" | "warn" | "fail" | "skip"
    message: str
    detail: Optional[str] = None


# ----------------------------------
# Individual checks
# --------------------------------------------

def _check_aggregate_not_all_null(ctx):
    r = ctx.analysis_result
    if not isinstance(r, (pd.DataFrame, pd.Series)):
        return CheckResult("Aggregate not all-NULL", "skip", "not tabular")
    intent = (ctx.query_understanding or {}).get("intent", "")
    if "aggreg" not in intent and "sum" not in (ctx.analysis_code or "").lower():
        return CheckResult("Aggregate not all-NULL", "skip", "not aggregate")
    if isinstance(r, pd.DataFrame):
        all_null = r.isna().all().all() or len(r) == 0
    else:
        all_null = r.isna().all() or len(r) == 0
    if all_null:
        return CheckResult("Aggregate not all-NULL", "fail",
                            "Aggregate returned no rows or all-NULL - "
                            "check column names and join conditions")
    return CheckResult("Aggregate not all-NULL", "pass", "has values")

def _check_no_error(ctx: AnalysisContext) -> CheckResult:
    if ctx.analysis_error:
        return CheckResult("Code executed cleanly", "fail",
                           "Analysis code raised an exception",
                           detail=ctx.analysis_error[-800:])
    return CheckResult("Code executed cleanly", "pass",
                       "No exceptions during execution")


def _check_result_present(ctx: AnalysisContext) -> CheckResult:
    if ctx.analysis_error:
        return CheckResult("Result variable produced", "skip",
                           "Skipped because code errored")
    if ctx.analysis_result is None:
        return CheckResult("Result variable produced", "fail",
                           "`result` is None - code may not have assigned it")
    return CheckResult("Result variable produced", "pass",
                       f"Got {type(ctx.analysis_result).__name__}")


def _check_result_nonempty(ctx: AnalysisContext) -> CheckResult:
    r = ctx.analysis_result
    if r is None:
        return CheckResult("Result is non-empty", "skip", "no result")
    if isinstance(r, (pd.DataFrame, pd.Series)):
        if len(r) == 0:
            return CheckResult("Result is non-empty", "warn",
                               "Result has zero rows - filter may be too strict")
        return CheckResult("Result is non-empty", "pass",
                           f"{len(r)} rows")
    if isinstance(r, (list, tuple, dict, np.ndarray)):
        n = len(r)
        if n == 0:
            return CheckResult("Result is non-empty", "warn", "Empty container")
        return CheckResult("Result is non-empty", "pass", f"{n} items")
    return CheckResult("Result is non-empty", "pass",
                       f"scalar of type {type(r).__name__}")


def _check_columns_referenced_exist(ctx: AnalysisContext) -> CheckResult:
    """Heuristic: column names quoted in the code must exist in some table."""
    print("FIX_A_LOADED")
    if not ctx.analysis_code:
        return CheckResult("Referenced columns exist", "skip", "no code")
    code = ctx.analysis_code

    available_cols: set = set()
    if ctx.store is not None:
        for tm in ctx.store.list_tables():
            for c, _ in tm.columns:
                available_cols.add(str(c))
    elif ctx.df is not None:
        available_cols = set(map(str, ctx.df.columns))
    else:
        return CheckResult("Referenced columns exist", "skip", "no schema")

    referenced = set()
    for m in re.finditer(r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code):
        referenced.add(m.group(1))
    for m in re.finditer(
            r"dfs\s*\[\s*['\"][^'\"]+['\"]\s*\]\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code):
        referenced.add(m.group(1))
    for m in re.finditer(r"\.groupby\(\s*['\"]([^'\"]+)['\"]\s*\)", code):
        referenced.add(m.group(1))

    # Exclude names that are CREATED by the code, not read from input.
    # These appear as the *target* of rename/assign/rename_axis/reset_index.
    created = set()
    for m in re.finditer(r"\.rename_axis\(\s*['\"]([^'\"]+)['\"]", code):
        created.add(m.group(1))
    for m in re.finditer(r"\.reset_index\([^)]*name\s*=\s*['\"]([^'\"]+)['\"]",
                         code):
        created.add(m.group(1))
    # rename(columns={"old": "new"}) - the new names are created
    for m in re.finditer(r"['\"][^'\"]+['\"]\s*:\s*['\"]([^'\"]+)['\"]", code):
        created.add(m.group(1))
    # .assign(new_col=...) - assignments using kwargs create new columns
    for m in re.finditer(r"\.assign\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", code):
        created.add(m.group(1))
    # `df['new_col'] = ...` - the LHS creates a new column.
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=(?!=)",
                     code):
        created.add(m.group(1))

    referenced -= created
    missing = referenced - available_cols
    if missing:
        return CheckResult(
            "Referenced columns exist", "fail",
            f"Code references columns not in any table: {sorted(missing)}"
        )
    # if missing:
    #     return CheckResult(
    #         "Referenced columns exist", "warn",  # was "fail"
    #         f"Code may reference columns not in any input table: {sorted(missing)}. "
    #         f"This often means the columns are created by the code (rename/assign/etc.) "
    #         f"rather than read from input - usually safe to ignore."
    #     )
    if not referenced:
        return CheckResult("Referenced columns exist", "skip",
                           "No explicit column references found in code")
    return CheckResult("Referenced columns exist", "pass",
                       f"All {len(referenced)} referenced columns present")

def _check_referenced_tables_exist(ctx: AnalysisContext) -> CheckResult:
    """Tables cited in pandas code (dfs['name']) or SQL FROM clauses exist."""
    code = (ctx.analysis_code or "")
    if not code or ctx.store is None:
        return CheckResult("Referenced tables exist", "skip", "no store")
    available = {tm.name for tm in ctx.store.list_tables()}

    referenced = set()
    for m in re.finditer(r"dfs\s*\[\s*['\"]([^'\"]+)['\"]\s*\]", code):
        referenced.add(m.group(1))
    # SQL FROM/JOIN identifiers
    if ctx.engine == "sql":
        for m in re.finditer(r'(?i)\b(?:from|join)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?',
                             code):
            referenced.add(m.group(1))

    missing = referenced - available
    if missing:
        return CheckResult("Referenced tables exist", "fail",
                           f"Code references tables that don't exist: {sorted(missing)}")
    if not referenced:
        return CheckResult("Referenced tables exist", "skip",
                           "No explicit table references")
    return CheckResult("Referenced tables exist", "pass",
                       f"All {len(referenced)} referenced tables present")


def _check_no_unexpected_nans(ctx: AnalysisContext) -> CheckResult:
    """If input had no NaN in numeric columns, result probably shouldn't either."""
    r = ctx.analysis_result
    if not isinstance(r, (pd.DataFrame, pd.Series)):
        return CheckResult("No surprise NaNs in result", "skip",
                           "Result is not a DataFrame/Series")
    # Resolve "input" - single df, or fall back to skip in multi-table case
    # since "did the input have NaNs" needs a join-aware answer we don't have.
    if ctx.df is not None:
        input_df = ctx.df
    elif ctx.store is not None and len(ctx.store.list_tables()) == 1:
        only = ctx.store.list_tables()[0]
        input_df = ctx.store.preview(only.name, n=1000)  # sample
    else:
        return CheckResult("No surprise NaNs in result", "skip",
                           "Multi-table input - NaN baseline ambiguous")

    input_num = input_df.select_dtypes(include="number")
    input_had_nan = input_num.isna().any().any() if not input_num.empty else False
    if isinstance(r, pd.DataFrame):
        result_num = r.select_dtypes(include="number")
        result_has_nan = result_num.isna().any().any() if not result_num.empty else False
    else:
        result_has_nan = r.isna().any() if pd.api.types.is_numeric_dtype(r) else False
    if result_has_nan and not input_had_nan:
        return CheckResult("No surprise NaNs in result", "warn",
                           "Result contains NaN but input numeric columns had none - "
                           "groupby may have introduced gaps")
    return CheckResult("No surprise NaNs in result", "pass",
                       "No unexpected NaNs introduced")


def _check_row_count_sane(ctx: AnalysisContext) -> CheckResult:
    """Aggregations should not return more rows than the input."""
    r = ctx.analysis_result
    if not isinstance(r, (pd.DataFrame, pd.Series)):
        return CheckResult("Row count plausible", "skip", "Result not tabular")
    intent = (ctx.query_understanding or {}).get("intent", "")
    if "aggreg" in intent or "groupby" in intent or "describe" in intent:
        # Determine the relevant input row count
        if ctx.df is not None:
            input_n = len(ctx.df)
        elif ctx.store is not None:
            tables_used = (ctx.query_understanding or {}).get("tables_needed") or [
                tm.name for tm in ctx.store.list_tables()]
            input_n = max(
                (ctx.store.schema(t).n_rows for t in tables_used if t in
                 {tm.name for tm in ctx.store.list_tables()}),
                default=0,
            )
        else:
            return CheckResult("Row count plausible", "skip", "no input baseline")
        if len(r) > input_n:
            return CheckResult("Row count plausible", "fail",
                               f"Aggregation returned {len(r)} rows from "
                               f"{input_n} input rows")
        return CheckResult("Row count plausible", "pass",
                           f"{len(r)} rows from {input_n} input rows")
    return CheckResult("Row count plausible", "skip",
                       f"Intent {intent!r} is not aggregation-shaped")


def _check_groupby_total_matches(ctx: AnalysisContext) -> CheckResult:
    """
    For sum/total aggregations: the sum across groups should equal the
    sum of the underlying column. We try this only when we can detect
    a clean groupby+sum pattern, AND we have a single source df.
    """
    code = (ctx.analysis_code or "").lower()
    if "groupby" not in code or ".sum(" not in code:
        return CheckResult("Groupby totals reconcile", "skip",
                           "Not a groupby+sum pattern")
    r = ctx.analysis_result
    if not isinstance(r, (pd.DataFrame, pd.Series)):
        return CheckResult("Groupby totals reconcile", "skip", "Result not tabular")

    # Only attempt when there's exactly one input table - otherwise we can't
    # cleanly attribute a sum back to a single source.
    if ctx.df is not None:
        source = ctx.df
    elif ctx.store is not None and len(ctx.store.list_tables()) == 1:
        only = ctx.store.list_tables()[0]
        source = ctx.store.query_pandas(f'SELECT * FROM "{only.name}"')
    else:
        return CheckResult("Groupby totals reconcile", "skip",
                           "Multi-table aggregation - can't reconcile cleanly")

    if isinstance(r, pd.Series):
        col = r.name
        if col is None or col not in source.columns:
            return CheckResult("Groupby totals reconcile", "skip",
                               "Cannot map result Series back to a source column")
        expected = float(source[col].sum())
        got = float(r.sum())
        if not np.isclose(got, expected, rtol=1e-6):
            return CheckResult("Groupby totals reconcile", "fail",
                               f"Group sum {got:.4g} != input sum {expected:.4g}")
        return CheckResult("Groupby totals reconcile", "pass",
                           f"Group sum {got:.4g} matches input total")
    mismatches = []
    for col in r.select_dtypes(include="number").columns:
        if col in source.columns and pd.api.types.is_numeric_dtype(source[col]):
            expected = float(source[col].sum())
            got = float(r[col].sum())
            if not np.isclose(got, expected, rtol=1e-6):
                mismatches.append((col, got, expected))
    if mismatches:
        msg = "; ".join(f"{c}: {g:.4g} vs {e:.4g}" for c, g, e in mismatches)
        return CheckResult("Groupby totals reconcile", "warn",
                           f"Some columns don't reconcile: {msg}")
    return CheckResult("Groupby totals reconcile", "pass",
                       "All reconcilable numeric columns match input totals")


# ----------------------------------------
# Public entry points
# ---------------------------------------

CHECKS = [
    _check_no_error,
    _check_result_present,
    _check_result_nonempty,
    _check_referenced_tables_exist,
    _check_columns_referenced_exist,
    _check_no_unexpected_nans,
    _check_row_count_sane,
    _check_groupby_total_matches,
    _check_aggregate_not_all_null
]


def run_auto_checks(ctx: AnalysisContext) -> list[CheckResult]:
    out: list[CheckResult] = []
    for fn in CHECKS:
        try:
            out.append(fn(ctx))
        except Exception as e:
            out.append(CheckResult(fn.__name__, "warn",
                                   f"Check itself errored: {e}"))
    return out


# ---------------------------------------------
# On-demand verification actions
# ---------------------------------------------

def compute_describe_baseline(df_or_ctx) -> "pd.DataFrame | dict":
    """
    Plain `df.describe(include='all')` - a trustworthy reference.

    Accepts either a DataFrame (legacy) or an AnalysisContext (multi-table
    aware). For multi-table, returns a dict mapping table_name -> describe.
    """
    if isinstance(df_or_ctx, pd.DataFrame):
        return df_or_ctx.describe(include="all")
    # AnalysisContext-like
    ctx = df_or_ctx
    if getattr(ctx, "df", None) is not None:
        return ctx.df.describe(include="all")
    if getattr(ctx, "store", None) is not None:
        return {tm.name: ctx.store.preview(tm.name, n=10000).describe(include="all")
                for tm in ctx.store.list_tables()}
    raise ValueError("no input data on context")


def recompute_with_sample(ctx: AnalysisContext, frac: float = 0.1,
                          random_state: int = 0) -> tuple[Any, Optional[str]]:
    """
    Re-run the same analysis code on a random sample of rows.

    Single-table & pandas: sample df, run code with df=sample.
    Multi-table & pandas:  sample EVERY input table, rebuild dfs, run code.
    SQL engine:            create a temp in-memory SQLite with sampled tables,
                           then re-run the original SQL against it.
    """
    import sqlite3

    if not ctx.analysis_code:
        return None, "no analysis code"

    if ctx.store is None and ctx.df is None:
        return None, "no input data available"

    if ctx.engine == "sql":
        if ctx.store is None:
            return None, "SQL engine requires a store"
        con = sqlite3.connect(":memory:")
        try:
            for tm in ctx.store.list_tables():
                full = ctx.store.query_pandas(f'SELECT * FROM "{tm.name}"')
                sampled = full.sample(frac=frac, random_state=random_state)
                sampled.to_sql(tm.name, con, index=False, if_exists="replace")
            result = pd.read_sql(ctx.analysis_code, con)
            return result, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        finally:
            con.close()

    ns: dict = {"__builtins__": __builtins__}
    ns["pd"] = pd
    ns["np"] = np

    if ctx.df is not None:
        ns["df"] = ctx.df.sample(frac=frac, random_state=random_state)
        ns["dfs"] = {"data": ns["df"]}
    elif ctx.store is not None:
        sampled = {}
        for tm in ctx.store.list_tables():
            full = ctx.store.query_pandas(f'SELECT * FROM "{tm.name}"')
            sampled[tm.name] = full.sample(frac=frac, random_state=random_state)
        ns["dfs"] = sampled
        if len(sampled) == 1:
            ns["df"] = next(iter(sampled.values()))
    else:
        return None, "no input data available"

    try:
        exec(compile(ctx.analysis_code, "<sample-rerun>", "exec"), ns)
        return ns.get("result"), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def code_in_plain_english_prompt(ctx: AnalysisContext) -> tuple[str, str]:
    """
    Returns (system, user) prompts you can send to an LLM to get a
    plain-English explanation of what the analysis code does.
    """
    system = ("You translate pandas code into plain-English step-by-step "
              "explanations. Be specific about column names and operations. "
              "5-8 short bullets, no preamble.")
    user = (f"User question: {ctx.user_query}\n\n"
            f"Code that ran:\n```python\n{ctx.analysis_code}\n```")
    return system, user