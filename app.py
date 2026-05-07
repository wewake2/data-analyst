"""
Streamlit chat frontend for the multi-agent data analyst (multi-table edition).

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
import time
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st

from data_analyst_agent import LLMConfig, Orchestrator, TableStore
from data_analyst_agent.core.logging_util import ring_buffer, configure_logging
from data_analyst_agent.core.relationships import (
    RelationshipCandidate, discover_relationships,
)
from data_analyst_agent.core.table_store import slugify_table_name
from data_analyst_agent.core.verification import (
    compute_describe_baseline, recompute_with_sample,
)

configure_logging()


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GetDelly Data Analyst",
    layout="wide",
)

# PROVIDERS = ["bedrock", "anthropic", "openai", "mock"]
PROVIDERS = ["bedrock" ]

# BEDROCK_MODEL_PRESETS = [
#     "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
#     "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
#     "anthropic.claude-haiku-4-5-20251001-v1:0",
#     "nvidia.nemotron-super-3-120b",
#     "nvidia.nemotron-nano-3-30b",
# ]
BEDROCK_MODEL_PRESETS = [
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
]

AGENT_NAMES = ["data_insight", "code_generation", "execution", "reasoning"]
STATUS_BADGE = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "store" not in st.session_state:
        st.session_state.store = None              # TableStore
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()    # filenames already loaded
    if "candidates" not in st.session_state:
        st.session_state.candidates = []           # list[RelationshipCandidate]
    if "confirmed_idx" not in st.session_state:
        st.session_state.confirmed_idx = set()     # indices of confirmed candidates
    if "rejected_idx" not in st.session_state:
        st.session_state.rejected_idx = set()


_init_state()


# -----------------------------------------------------------------------------
# Sidebar: LLM configuration
# -----------------------------------------------------------------------------
def _llm_config_widget(label: str, key_prefix: str,
                       default_provider: str = "bedrock") -> LLMConfig:
    provider = st.selectbox(
        f"{label} - provider", PROVIDERS,
        index=PROVIDERS.index(default_provider),
        key=f"{key_prefix}_provider",
    )
    if provider == "bedrock":
        model = st.text_input(f"{label} - model id",
                              value=BEDROCK_MODEL_PRESETS[0],
                              key=f"{key_prefix}_model")
        with st.expander(f"{label} - presets"):
            for preset in BEDROCK_MODEL_PRESETS:
                if st.button(preset, key=f"{key_prefix}_preset_{preset}"):
                    st.session_state[f"{key_prefix}_model"] = preset
                    st.rerun()
    elif provider == "anthropic":
        model = st.text_input(f"{label} - model", value="claude-sonnet-4-5",
                              key=f"{key_prefix}_model")
    elif provider == "openai":
        model = st.text_input(f"{label} - model", value="gpt-4o",
                              key=f"{key_prefix}_model")
    else:
        model = "mock"

    temp = st.slider(f"{label} - temperature", 0.0, 1.0, 0.2, 0.05,
                     key=f"{key_prefix}_temp")
    max_tokens = st.number_input(f"{label} - max_tokens",
                                 min_value=256, max_value=8192, value=2048, step=256,
                                 key=f"{key_prefix}_maxtok")

    extra: dict = {}
    if provider == "bedrock":
        region = st.text_input(f"{label} - AWS region",
                               value=os.getenv("AWS_REGION", "us-east-1"),
                               key=f"{key_prefix}_region")
        extra["aws_region"] = region
        if "nemotron" in model.lower():
            extra["enable_thinking"] = st.checkbox(
                f"{label} - Nemotron thinking on", value=False,
                key=f"{key_prefix}_thinking",
            )

    return LLMConfig(provider=provider, model=model, temperature=temp,
                     max_tokens=int(max_tokens), extra=extra)


with st.sidebar:
    st.title("Configuration")
    st.markdown("### Default LLM")
    default_cfg = _llm_config_widget("Default", "default")

    st.divider()
    st.markdown("### Per-agent overrides")
    overrides: dict[str, LLMConfig] = {}
    for agent in AGENT_NAMES:
        if agent == "execution":
            with st.expander(f"`{agent}` (no LLM - runs code)"):
                st.caption("Execution agent runs code in a sandboxed namespace.")
            continue
        on = st.toggle(f"Override `{agent}`", value=False, key=f"ovr_{agent}")
        if on:
            with st.expander(f"`{agent}` settings", expanded=True):
                overrides[agent] = _llm_config_widget(agent, f"ag_{agent}")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Reset entire session"):
        # Drop store + chat
        if st.session_state.store is not None:
            st.session_state.store.close()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.title("GetDelly Data Analyst")
st.caption("Upload one or more CSVs. Relationships are auto-detected; "
           "confirm them before joining.")

uploads = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

# Ingest any newly uploaded files
new_files = []
if uploads:
    if st.session_state.store is None:
        st.session_state.store = TableStore()
    for upl in uploads:
        if upl.name in st.session_state.ingested_files:
            continue
        new_files.append(upl)

if new_files:
    st.warning(f"Ingesting {len(new_files)} file(s). Larger CSVs take longer ")
    overall = st.progress(0.0, "starting…")
    for i, upl in enumerate(new_files):
        per_file = st.progress(0.0, f"{upl.name} - opening…")
        # Stream the upload to a tempfile so we don't hold a 500MB bytes blob.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            CHUNK = 8 * 1024 * 1024
            while True:
                buf = upl.read(CHUNK)
                if not buf:
                    break
                tmp.write(buf)
            tmp_path = tmp.name
        try:
            slug = slugify_table_name(
                upl.name,
                existing={tm.name for tm in st.session_state.store.list_tables()},
            )
            def _cb(frac, msg, _bar=per_file, _name=upl.name):
                _bar.progress(min(1.0, frac), f"{_name} - {msg}")
            st.session_state.store.ingest_csv(tmp_path, table_name=slug, progress_cb=_cb)
            st.session_state.ingested_files.add(upl.name)
        finally:
            os.unlink(tmp_path)
        overall.progress((i + 1) / len(new_files),
                         f"ingested {i + 1}/{len(new_files)}")
    overall.progress(1.0, "done")
    # Re-discover relationships after every change
    if st.session_state.store is not None:
        st.session_state.candidates = discover_relationships(st.session_state.store)
        st.session_state.confirmed_idx = set()
        st.session_state.rejected_idx = set()
    st.success(f"Loaded {len(new_files)} file(s).")
    time.sleep(0.5)
    st.rerun()


# -----------------------------------------------------------------------------
# Tables panel
# -----------------------------------------------------------------------------
store: TableStore | None = st.session_state.store
if store is not None and store.list_tables():
    with st.expander(f"Loaded tables ({len(store.list_tables())})", expanded=True):
        for tm in store.list_tables():
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.markdown(f"**`{tm.name}`** ← {tm.source_filename}")
            with cols[1]:
                st.metric("Rows", f"{tm.n_rows:,}")
            with cols[2]:
                st.metric("Cols", tm.n_cols)
            with cols[3]:
                if st.button("Preview", key=f"prev_{tm.name}"):
                    st.session_state[f"_show_prev_{tm.name}"] = True
            if st.session_state.get(f"_show_prev_{tm.name}"):
                st.dataframe(store.preview(tm.name, n=20), use_container_width=True)


# -----------------------------------------------------------------------------
# Relationships panel
# -----------------------------------------------------------------------------
def _confirmed_candidates() -> list[RelationshipCandidate]:
    return [c for i, c in enumerate(st.session_state.candidates)
            if i in st.session_state.confirmed_idx]


if store is not None and len(store.list_tables()) > 1:
    cands = st.session_state.candidates
    with st.expander(f"Discovered relationships ({len(cands)} candidates, "
                     f"{len(_confirmed_candidates())} confirmed)",
                     expanded=True):
        if not cands:
            st.caption("No relationships detected. Tables will be queried independently.")
        else:
            st.caption("These were detected automatically by checking value overlap "
                       "and column-name patterns. Confirm the ones you want the agent "
                       "to use as join keys.")
            if st.button("Confirm all", key="confirm_all_rels"):
                for i, c in enumerate(cands):
                    if i not in st.session_state.confirmed_idx and i not in st.session_state.rejected_idx:
                        st.session_state.confirmed_idx.add(i)
                        try:
                            store.register_foreign_key(
                                c.child_table, c.child_col,
                                c.parent_table, c.parent_col,
                            )
                        except Exception:
                            pass
                st.rerun()
            for i, c in enumerate(cands):
                if i in st.session_state.rejected_idx:
                    continue
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    confirmed = i in st.session_state.confirmed_idx
                    label = "[confirmed] " if confirmed else ""
                    st.markdown(f"{label}**{c.label()}**  "
                                f"<small>conf={c.confidence:.2f} · kind={c.kind}</small>",
                                unsafe_allow_html=True)
                    with st.expander("evidence"):
                        st.json(c.evidence)
                with cols[1]:
                    if i not in st.session_state.confirmed_idx:
                        if st.button("Confirm", key=f"conf_{i}"):
                            st.session_state.confirmed_idx.add(i)
                            try:
                                store.register_foreign_key(
                                    c.child_table, c.child_col,
                                    c.parent_table, c.parent_col,
                                )
                            except Exception as e:
                                st.error(f"FK registration failed: {e}")
                            st.rerun()
                    else:
                        st.caption("confirmed")
                with cols[2]:
                    if st.button("Reject", key=f"rej_{i}"):
                        st.session_state.rejected_idx.add(i)
                        st.session_state.confirmed_idx.discard(i)
                        st.rerun()


# -----------------------------------------------------------------------------
# Render helpers (same shape as before)
# -----------------------------------------------------------------------------
def _format_evidence(rc) -> str:
    ref = rc.evidence_ref or {}
    kind = rc.evidence_kind
    if kind == "cell":  return f"cell at row=`{ref.get('row')}`, col=`{ref.get('col')}`"
    if kind == "row":   return f"row=`{ref.get('row')}`"
    if kind == "column": return f"column=`{ref.get('col')}`"
    if kind == "stat":  return f"`{ref.get('stat')}` of column `{ref.get('col')}`"
    if kind == "shape": return "result shape"
    if kind == "scalar": return f"scalar value `{ref.get('value')}`"
    return f"{kind} {ref}"


def _render_provenance(ctx) -> None:
    if not getattr(ctx, "resolved_claims", None):
        return
    st.markdown("##### Citations")
    for rc in ctx.resolved_claims:
        badge = STATUS_BADGE.get(rc.status, "?")
        st.markdown(f"{badge} {rc.text}  \n"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;<small>↳ {_format_evidence(rc)}</small>",
                    unsafe_allow_html=True)
        if rc.status != "pass":
            with st.expander("see why this didn't ground"):
                if rc.resolution_error:
                    st.error(f"Could not resolve evidence: {rc.resolution_error}")
                else:
                    st.warning(rc.grounding_detail)


def _render_value(v) -> None:
    if isinstance(v, pd.DataFrame):
        st.dataframe(v.head(50), use_container_width=True)
    elif isinstance(v, pd.Series):
        st.dataframe(v.head(50).to_frame(), use_container_width=True)
    elif v is None:
        st.caption("None")
    else:
        st.write(v)


def _render_verification_summary(ctx) -> None:
    if not getattr(ctx, "checks", None):
        return
    n_pass = sum(1 for c in ctx.checks if c.status == "pass")
    n_warn = sum(1 for c in ctx.checks if c.status == "warn")
    n_fail = sum(1 for c in ctx.checks if c.status == "fail")
    parts = []
    if n_fail: parts.append(f"{n_fail} failed")
    if n_warn: parts.append(f"{n_warn} warnings")
    if n_pass: parts.append(f"{n_pass} passed")
    summary = " · ".join(parts) if parts else "no checks"
    if n_fail: st.error(f"Verification: {summary}")
    elif n_warn: st.warning(f"Verification: {summary}")
    else: st.success(f"Verification: {summary}")


def _render_verification_tab(ctx, msg_idx: int) -> None:
    st.markdown("##### Automatic checks")
    if not getattr(ctx, "checks", None):
        st.caption("No checks were run.")
    else:
        for c in ctx.checks:
            badge = STATUS_BADGE.get(c.status, "?")
            st.markdown(f"{badge} **{c.name}** - {c.message}")
            if c.detail:
                with st.expander("details"):
                    st.code(c.detail, language="text")

    st.divider()
    st.markdown("##### Verify it yourself")
    describe_key = f"describe_{msg_idx}"
    sample_key   = f"sample_{msg_idx}"
    counts_key   = f"counts_{msg_idx}"
    col1, col2, col3 = st.columns(3)
    # TODO : fix these, not working
    with col1:
        if st.button("Independent describe()", key=f"btn_{describe_key}"):
            try:
                st.session_state[describe_key] = compute_describe_baseline(ctx)
            except Exception as e:
                st.session_state[describe_key] = e
    with col2:
        if st.button("Re-run on 10% sample", key=f"btn_{sample_key}"):
            st.session_state[sample_key] = recompute_with_sample(ctx, frac=0.1)
    with col3:
        if st.button("Show row counts", key=f"btn_{counts_key}"):
            st.session_state[counts_key] = True

    val = st.session_state.get(describe_key)
    if isinstance(val, pd.DataFrame):
        st.markdown("**df.describe(include='all')** - independent baseline")
        st.dataframe(val, use_container_width=True)
    elif isinstance(val, dict):
        for tname, dfv in val.items():
            st.markdown(f"**{tname}.describe()**")
            st.dataframe(dfv, use_container_width=True)
    elif isinstance(val, Exception):
        st.error(f"describe() failed: {val}")

    val = st.session_state.get(sample_key)
    if isinstance(val, tuple) and len(val) == 2:
        sample_result, err = val
        st.markdown("**Re-run on 10% sample**")
        if err:
            st.error(err)
        else:
            full = ctx.analysis_result
            cs = st.columns(2)
            with cs[0]:
                st.caption("Original (full data)")
                _render_value(full)
            with cs[1]:
                st.caption("Sample re-run (10%)")
                _render_value(sample_result)

    if st.session_state.get(counts_key) is True:
        if ctx.df is not None:
            df = ctx.df
            st.markdown("**Input dataframe**")
            info = pd.DataFrame({
                "dtype": df.dtypes.astype(str),
                "non_null": df.notna().sum(),
                "null": df.isna().sum(),
                "n_unique": df.nunique(dropna=True),
            })
            st.dataframe(info, use_container_width=True)
            st.caption(f"Total: {len(df):,} rows × {len(df.columns)} cols")
        elif ctx.store is not None:
            for tm in ctx.store.list_tables():
                st.markdown(f"**`{tm.name}`** - {tm.n_rows:,} rows × {tm.n_cols} cols")


def _render_logs_tab(ctx) -> None:
    records = ring_buffer.snapshot()
    started_at = getattr(ctx, "_started_at", None)
    if started_at is not None:
        records = [r for r in records if r.ts >= started_at]
    if not records:
        st.caption("No log records.")
        return
    levels = st.multiselect("Levels", ["DEBUG", "INFO", "WARNING", "ERROR"],
                            default=["INFO", "WARNING", "ERROR"],
                            key=f"loglevel_{id(ctx)}")
    show_payload = st.checkbox("Show payloads", value=False,
                               key=f"logpayload_{id(ctx)}")
    for r in records:
        if r.level not in levels:
            continue
        ts = datetime.fromtimestamp(r.ts).strftime("%H:%M:%S.%f")[:-3]
        dur = f"  ({r.duration_ms:.0f} ms)" if r.duration_ms else ""
        prefix = {"INFO": "[INFO]", "WARNING": "[WARN]", "ERROR": "[ERROR]", "DEBUG": "[DEBUG]"}.get(r.level, "-")
        st.markdown(f"`{ts}` {prefix} **{r.agent}** - {r.message}{dur}")
        if show_payload and r.payload:
            with st.expander("payload"):
                st.code(r.payload[:5000] +
                        ("\n... (truncated)" if len(r.payload) > 5000 else ""),
                        language="text")


def _render_provenance_tab(ctx) -> None:
    if not getattr(ctx, "resolved_claims", None):
        st.caption("No structured claims emitted.")
        return
    n_pass = sum(1 for c in ctx.resolved_claims if c.status == "pass")
    n_warn = sum(1 for c in ctx.resolved_claims if c.status == "warn")
    n_fail = sum(1 for c in ctx.resolved_claims if c.status == "fail")
    total = len(ctx.resolved_claims)
    if n_fail:
        st.error(f"{n_fail}/{total} claim(s) had unresolvable references.")
    elif n_warn:
        st.warning(f"{n_warn}/{total} claim(s) cite numbers not in the result.")
    else:
        st.success(f"All {total} claim(s) ground to verified values.")
    st.markdown("---")
    for i, rc in enumerate(ctx.resolved_claims, 1):
        badge = STATUS_BADGE.get(rc.status, "?")
        st.markdown(f"**{i}. {badge} {rc.text}**")
        st.markdown(f"<small>↳ {_format_evidence(rc)}</small>",
                    unsafe_allow_html=True)
        if isinstance(rc.resolved_value, pd.Series):
            st.dataframe(rc.resolved_value.to_frame(), use_container_width=True)
        elif isinstance(rc.resolved_value, pd.DataFrame):
            st.dataframe(rc.resolved_value, use_container_width=True)
        elif rc.resolution_error:
            st.error(rc.resolution_error)
        else:
            st.code(str(rc.resolved_value), language="text")
        st.caption(f"_{rc.grounding_detail or 'no grounding info'}_")


def _render_analysis(ctx, msg_idx: int) -> None:
    if ctx.explanation:
        st.markdown(ctx.explanation)

    has_plot = ctx.plot_path and os.path.exists(ctx.plot_path)
    has_data = (ctx.analysis_result is not None) and not ctx.analysis_error
    if has_plot or has_data:
        result_tabs = st.tabs(["Plot", "Data"])
        with result_tabs[0]:
            if has_plot:
                st.image(ctx.plot_path, use_container_width=True)
            else:
                st.caption("No plot was generated.")
        with result_tabs[1]:
            if has_data:
                r = ctx.analysis_result
                if isinstance(r, pd.DataFrame):
                    st.dataframe(r.head(500), use_container_width=True)
                elif isinstance(r, pd.Series):
                    st.dataframe(r.head(500).to_frame(), use_container_width=True)
                else:
                    st.write(r)
            else:
                st.caption("No tabular data available.")

    _render_provenance(ctx)
    _render_verification_summary(ctx)

    if ctx.token_usage.total > 0:
        st.caption(
            f"Tokens: {ctx.token_usage.input_tokens:,} input / "
            f"{ctx.token_usage.output_tokens:,} output / "
            f"{ctx.token_usage.total:,} total"
        )

    if ctx.analysis_error:
        st.error("Analysis code raised an exception - see Result tab.")
    if ctx.plot_error:
        st.warning("Plot code failed - see trace.")

    verify_active = any(
        st.session_state.get(f"{k}_{msg_idx}") is not None
        for k in ("describe", "sample", "counts")
    )
    with st.expander("Per-agent breakdown", expanded=verify_active):
        tabs = st.tabs(["Insight", "Plan", "Code", "Result",
                        "Verify", "Provenance", "Logs", "Trace"])
        with tabs[0]:
            st.json(ctx.df_summary or {})
        with tabs[1]:
            plan = ctx.query_understanding or {}
            if plan:
                st.markdown(f"**Engine**: `{ctx.engine}`")
                if plan.get("tables_needed"):
                    st.markdown(f"**Tables**: {plan['tables_needed']}")
                if plan.get("joins"):
                    st.markdown("**Joins**:")
                    st.json(plan["joins"])
            st.json(plan)
        with tabs[2]:
            if ctx.analysis_code:
                st.markdown(f"**Analysis code ({ctx.engine})**")
                st.code(ctx.analysis_code,
                        language="sql" if ctx.engine == "sql" else "python")
            if ctx.plot_code:
                st.markdown("**Plot code**")
                st.code(ctx.plot_code, language="python")
        with tabs[3]:
            if ctx.analysis_error:
                st.code(ctx.analysis_error, language="text")
            else:
                r = ctx.analysis_result
                if isinstance(r, pd.DataFrame):
                    st.markdown(f"**Result** ({len(r):,} rows × {len(r.columns)} cols)")
                    st.dataframe(r.head(500), use_container_width=True)
                elif isinstance(r, pd.Series):
                    st.markdown(f"**Result Series** ({len(r):,} entries)")
                    st.dataframe(r.head(500).to_frame(), use_container_width=True)
                elif r is not None:
                    st.write(r)
                else:
                    st.caption("No `result` was produced.")
            if ctx.analysis_stdout:
                st.markdown("**stdout**")
                st.code(ctx.analysis_stdout, language="text")
        with tabs[4]: _render_verification_tab(ctx, msg_idx)
        with tabs[5]: _render_provenance_tab(ctx)
        with tabs[6]: _render_logs_tab(ctx)
        with tabs[7]:
            for entry in ctx.trace:
                st.markdown(f"- **[{entry['agent']}]** {entry['message']}")


# -----------------------------------------------------------------------------
# Render existing chat history
# -----------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("ctx") is not None:
            _render_analysis(msg["ctx"], msg_idx=i)
        else:
            st.markdown(msg["content"])


# -----------------------------------------------------------------------------
# Chat input
# -----------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about your data…")

if prompt:
    if store is None or not store.list_tables():
        st.error("Please upload at least one CSV first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            orch = Orchestrator(default_llm=default_cfg, agent_llms=overrides)
        except Exception as e:
            st.error(f"Could not build orchestrator: {e}")
            st.code(traceback.format_exc())
            st.stop()

        status = st.status("Running agents…", expanded=True)
        with status:
            t0 = time.time()
            try:
                ctx = orch.analyze_store(
                    store, prompt,
                    confirmed_relationships=_confirmed_candidates(),
                )
                ctx._started_at = t0
                status.update(label=f"Done in {time.time() - t0:.1f}s",
                              state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")
                st.code(traceback.format_exc())
                st.stop()

        msg_idx = len(st.session_state.messages)
        _render_analysis(ctx, msg_idx=msg_idx)
        st.session_state.messages.append({
            "role": "assistant",
            "content": ctx.explanation or "(no explanation produced)",
            "ctx": ctx,
        })