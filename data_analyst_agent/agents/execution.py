"""
ExecutionAgent
-----------------------------
Runs the analysis in one of two ways :
  - engine == "sql":     ctx.store.query_pandas(ctx.analysis_code)
  - engine == "pandas":  exec(ctx.analysis_code) in a sandbox namespace that has `df` (single-table) and `dfs` (multi-table)
Plot code always runs in pandas - matplotlib needs in-memory data anyway.
Same security caveat as before: this is a best-effort sandbox, not a
security boundary. Use only for code you'd be willing to run yourself.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

import pandas as pd

from ..core.agent_base import Agent
from ..core.context import AnalysisContext
from ..core.logging_util import get_logger, timed


FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "requests",
    "urllib", "http", "ftplib", "telnetlib", "pickle", "marshal",
    "ctypes", "multiprocessing", "threading",
}

SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "getattr", "hasattr", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "print", "range",
    "repr", "reversed", "round", "set", "setattr", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "__import__",
}


def _make_safe_import():
    real_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in FORBIDDEN_IMPORTS:
            raise ImportError(f"Import of '{name}' is not allowed in execution agent")
        return real_import(name, globals, locals, fromlist, level)
    return safe_import

def _build_namespace(ctx: AnalysisContext) -> dict:
    safe_builtins = {k: getattr(builtins, k) for k in SAFE_BUILTINS if hasattr(builtins, k)}
    safe_builtins["__import__"] = _make_safe_import()
    ns: dict = {"__builtins__": safe_builtins}

    if ctx.store is not None:
        ns["dfs"] = ctx.store.to_dataframes()
        # Expose `df` whenever the query plan needs exactly one table -
        # not just when only one table is required. This handles the common
        # case where the user asks a single-table question against a
        # multi-table store ("how many products per category?").
        plan_tables = (ctx.query_understanding or {}).get("tables_needed") or []
        if len(plan_tables) == 1 and plan_tables[0] in ns["dfs"]:
            ns["df"] = ns["dfs"][plan_tables[0]]
        elif len(ns["dfs"]) == 1:
            ns["df"] = next(iter(ns["dfs"].values()))
    elif ctx.df is not None:
        ns["df"] = ctx.df
        ns["dfs"] = {"data": ctx.df}
    return ns

class ExecutionAgent(Agent):
    name = "execution"
    system_prompt = ""

    def __init__(self, output_dir: str = "/tmp/data_analyst_outputs", **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._log = get_logger(self.name)

    def _run_analysis_sql(self, ctx: AnalysisContext) -> None:
        if not ctx.analysis_code:
            ctx.analysis_error = "No SQL generated"
            self._log.warning("no SQL to run")
            return
        if ctx.store is None:
            ctx.analysis_error = "SQL engine selected but no TableStore available"
            self._log.error(ctx.analysis_error)
            return
        self._log.info("about to run SQL", extra={"payload": ctx.analysis_code})
        try:
            with timed(self._log, "exec SQL"):
                ctx.analysis_result = ctx.store.query_pandas(ctx.analysis_code)
                self._log.info(
                    f"SQL DEBUG: db={ctx.store.db_path}  "
                    f"result_type={type(ctx.analysis_result).__name__}  "
                    f"shape={getattr(ctx.analysis_result, 'shape', '?')}  "
                    f"head:\n{ctx.analysis_result.head(3) if hasattr(ctx.analysis_result, 'head') else ctx.analysis_result!r}"
                )
            preview = self._preview(ctx.analysis_result)
            self._log.info(
                f"SQL result rows={len(ctx.analysis_result)} "
                f"cols={len(ctx.analysis_result.columns)}",
                extra={"payload": preview},
            )
            ctx.log(self.name, f"SQL ran. {len(ctx.analysis_result)} rows returned")
        except Exception:
            ctx.analysis_error = traceback.format_exc()
            self._log.error("SQL raised", extra={"payload": ctx.analysis_error})
            ctx.log(self.name, "SQL raised an exception")

    def _run_analysis_pandas(self, ctx: AnalysisContext) -> None:
        if not ctx.analysis_code:
            ctx.analysis_error = "No analysis code generated"
            self._log.warning("no analysis code to run")
            return
        self._log.info("about to run pandas code",
                       extra={"payload": ctx.analysis_code})
        ns = _build_namespace(ctx)
        stdout = io.StringIO()
        try:
            with timed(self._log, "exec pandas code"):
                with contextlib.redirect_stdout(stdout):
                    exec(compile(ctx.analysis_code, "<analysis>", "exec"), ns)
            ctx.analysis_result = ns.get("result")
            ctx.analysis_stdout = stdout.getvalue()
            preview = self._preview(ctx.analysis_result)
            self._log.info(
                f"result type={type(ctx.analysis_result).__name__}",
                extra={"payload": preview},
            )
            if ctx.analysis_stdout:
                self._log.info("stdout captured",
                               extra={"payload": ctx.analysis_stdout})
            ctx.log(self.name, f"Analysis ran. result type: "
                               f"{type(ctx.analysis_result).__name__}")
        except Exception:
            ctx.analysis_error = traceback.format_exc()
            self._log.error("analysis raised",
                            extra={"payload": ctx.analysis_error})
            ctx.log(self.name, "Analysis raised an exception")

    def _run_plot(self, ctx: AnalysisContext) -> None:
        if not ctx.plot_code:
            return
        self._log.info("about to run plot code", extra={"payload": ctx.plot_code})
        ns = _build_namespace(ctx)
        ns["result"] = ctx.analysis_result
        ns["plt"] = plt
        import pandas as _pd
        import numpy as _np
        ns["pd"] = _pd
        ns["np"] = _np
        try:
            with timed(self._log, "exec plot code"):
                exec(compile(ctx.plot_code, "<plot>", "exec"), ns)
            fig = ns.get("fig") or plt.gcf()
            path = os.path.join(self.output_dir, "plot.png")
            fig.savefig(path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            ctx.plot_path = path
            self._log.info(f"plot saved -> {path}")
            ctx.log(self.name, f"Plot saved -> {path}")
        except Exception:
            ctx.plot_error = traceback.format_exc()
            self._log.error("plot raised",
                            extra={"payload": ctx.plot_error})
            ctx.log(self.name, "Plot raised an exception")
            plt.close("all")

    @staticmethod
    def _preview(result, max_chars: int = 1500) -> str:
        try:
            s = result.to_string() if hasattr(result, "to_string") else repr(result)
        except Exception:
            s = repr(result)
        return s if len(s) <= max_chars else s[:max_chars] + "\n... (truncated)"

    def run(self, ctx: AnalysisContext) -> None:
        if ctx.engine == "sql":
            self._run_analysis_sql(ctx)
        else:
            self._run_analysis_pandas(ctx)
        self._run_plot(ctx)