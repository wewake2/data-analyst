"""
Provenance resolution.

For each {text, evidence_kind, evidence_ref} claim emitted by the reasoning agent :

1. Resolve the evidence_ref against ctx.analysis_result -> a concrete value (or an error if the path doesn't exist).
2. Extract numeric tokens from the claim's `text`.
3. Check that *every* number in the text matches *some* number derivable
   from the resolved evidence (the cell value, or any of the row's values, etc.) within a small relative tolerance.

A claim that doesn't ground (i.e. text mentions 87432 but the resolved cell holds 89001) is marked UNVERIFIED and the UI shows it with a red badge.

The resolver is intentionally tolerant about LLM-style ref shapes - index
labels can be ints or strings, columns are looked up case-insensitively as
a fallback, etc. Better to be lenient on the *path* and strict on the
*number match*.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ResolvedClaim:
    text: str
    evidence_kind: str
    evidence_ref: dict
    resolved_value: Any = None  # the actual data the ref pointed at
    resolution_error: Optional[str] = None
    grounded: bool = False  # numbers in text match resolved value
    grounding_detail: str = ""

    @property
    def status(self) -> str:
        if self.resolution_error:
            return "fail"  # ref doesn't exist
        if not self.grounded:
            return "warn"  # numbers don't match
        return "pass"


# --------------------------------------------------------------------------- #
# Step 1: resolve evidence_ref -> a value from the result
# --------------------------------------------------------------------------- #

def _lookup_column(result: pd.DataFrame, col: str) -> Optional[str]:
    """Return the actual column name if `col` matches, else None."""
    if col in result.columns:
        return col
    # Case-insensitive fallback
    lower = {str(c).lower(): c for c in result.columns}
    return lower.get(str(col).lower())


def _lookup_row(obj, row_key) -> Any:
    """Try .loc[row_key], then iloc[row_key] if int, then case-insensitive."""
    try:
        return obj.loc[row_key]
    except (KeyError, TypeError):
        pass
    if isinstance(row_key, int):
        try:
            return obj.iloc[row_key]
        except (IndexError, TypeError):
            pass
    # case-insensitive on object-typed indexes
    if hasattr(obj, "index"):
        idx_map = {str(i).lower(): i for i in obj.index}
        actual = idx_map.get(str(row_key).lower())
        if actual is not None:
            try:
                return obj.loc[actual]
            except (KeyError, TypeError):
                pass
    raise KeyError(row_key)


def resolve(result: Any, kind: str, ref: dict) -> Any:
    """Resolve an evidence ref against the result. Raises on failure."""
    if result is None:
        raise ValueError("result is None")

    if kind == "scalar":
        # The "value" itself is the evidence; we still verify it's
        # consistent with the result if the result is a scalar.
        return ref.get("value")

    if kind == "shape":
        if hasattr(result, "shape"):
            return result.shape
        return (len(result),) if hasattr(result, "__len__") else None

    if kind == "stat":
        col = ref.get("col")
        stat = ref.get("stat")
        # Resolve which Series to apply the stat to
        if isinstance(result, pd.Series):
            series = result  # col is irrelevant here
        elif isinstance(result, pd.DataFrame):
            if col is None:
                raise ValueError("stat ref on DataFrame requires 'col'")
            actual_col = _lookup_column(result, col)
            if actual_col is None:
                # Helpful error: did the LLM cite an *input* column?
                raise KeyError(
                    f"column {col!r} not in result. "
                    f"Result columns are: {list(result.columns)}. "
                    f"Note: this is the *result*, not the input table."
                )
            series = result[actual_col]
        else:
            raise TypeError(
                f"stat ref needs Series/DataFrame result, got {type(result).__name__}"
            )
        
        fn = {
            "min": series.min,
            "max": series.max,
            "mean": series.mean,
            "sum": series.sum,
            "count": series.count,
            "std": series.std,
            "median": series.median,
            "var": series.var,
        }.get(stat)
        if fn is None:
            raise ValueError(f"unknown stat {stat!r}")
        return fn()

    if kind == "column":
        col = ref.get("col")
        if isinstance(result, pd.DataFrame):
            actual_col = _lookup_column(result, col)
            if actual_col is None:
                raise KeyError(f"column {col!r} not in result")
            return result[actual_col]
        if isinstance(result, dict):
            # dict of {col -> Series/values}
            if col in result:
                return result[col]
            lower = {str(k).lower(): k for k in result.keys()}
            actual = lower.get(str(col).lower())
            if actual is not None:
                return result[actual]
            raise KeyError(f"column {col!r} not in result dict")
        raise TypeError(f"column ref needs DataFrame/dict, got {type(result).__name__}")

    if kind == "row":
        if isinstance(result, dict):
            # treat as scalar lookup
            if ref.get("row") in result:
                return result[ref.get("row")]
            raise KeyError(f"row {ref.get('row')!r} not in result dict")
        return _lookup_row(result, ref.get("row"))

    if kind == "cell":
        row = ref.get("row")
        col = ref.get("col")
        if isinstance(result, pd.Series):
            return _lookup_row(result, row)
        if isinstance(result, pd.DataFrame):
            actual_col = _lookup_column(result, col)
            if actual_col is None:
                raise KeyError(f"column {col!r} not in result")
            # 1: row is an index label
            try:
                row_obj = _lookup_row(result, row)
                if isinstance(row_obj, pd.Series) and actual_col in row_obj.index:
                    return row_obj[actual_col]
                return row_obj
            except KeyError:
                pass

            # 2: row is a VALUE in some other column. Common when
            # the LLM cites "Clothing" but the result has a 'category'
            # column rather than a 'category' index. Scan object/category
            # columns for an exact match (case-insensitive fallback).
            for candidate_col in result.columns:
                if candidate_col == actual_col:
                    continue
                series = result[candidate_col]
                if not (
                    series.dtype == object
                    or pd.api.types.is_string_dtype(series)
                    or pd.api.types.is_categorical_dtype(series)
                ):
                    continue
                # exact match
                hits = result[series == row]
                if len(hits) == 0:
                    # case-insensitive fallback
                    hits = result[series.astype(str).str.lower() == str(row).lower()]
                if len(hits) == 1:
                    return hits.iloc[0][actual_col]
                if len(hits) > 1:
                    raise KeyError(
                        f"row value {row!r} matched {len(hits)} rows in "
                        f"column {candidate_col!r} - ambiguous"
                    )

            raise KeyError(
                f"row {row!r} not found as index label or as a value "
                f"in any text column of result"
            )
        raise TypeError(
            f"cell ref needs Series/DataFrame/dict, got {type(result).__name__}"
        )

    raise ValueError(f"unknown evidence_kind {kind!r}")


# --------------------------------------------------------------------------- #
# Step 2: extract numbers from claim text and ground them
# --------------------------------------------------------------------------- #
# Matches ints, floats, and numbers with thousand separators or %, $, etc.
_NUM_RE = re.compile(r"[-+]?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+\.?\d*")


def _extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        cleaned = m.replace(",", "").replace("$", "").strip()
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out

def _values_from(resolved: Any) -> list[float]:
    """Collect candidate numeric values reachable from `resolved`."""
    out: list[float] = []
    if resolved is None:
        return out
    if isinstance(resolved, bool):  # bool is a subclass of int - exclude
        return out
    if isinstance(resolved, (int, float)):
        if not (isinstance(resolved, float) and math.isnan(resolved)):
            out.append(float(resolved))
        return out
    if isinstance(resolved, np.generic):
        try:
            v = float(resolved)
            if not math.isnan(v):
                out.append(v)
        except (TypeError, ValueError):
            pass
        return out
    if isinstance(resolved, str):
        # Strings can carry numbers (bin labels like "(0.0, 195.86]", currency strings, percentages). Extract them.
        out.extend(_extract_numbers(resolved))
        return out
    if isinstance(resolved, (tuple, list)):
        for x in resolved:
            out.extend(_values_from(x))
        return out
    if isinstance(resolved, pd.Series):
        for v in resolved.values:
            out.extend(_values_from(v))
        return out
    if isinstance(resolved, pd.DataFrame):
        for v in resolved.to_numpy().ravel():
            out.extend(_values_from(v))
        return out
    return out


def _matches(needle: float, haystack: list[float], rtol: float = 0.01) -> bool:
    """A number in the claim matches if it's close to any candidate value."""
    candidates = list(haystack)
    # Add format-equivalent forms: percentage <-> decimal. If haystack has 0.4691, also accept 46.91. If it has 46.91, also
    # accept 0.4691. Only when the value is in a plausible range for the conversion (avoid spurious matches).
    expanded = list(candidates)
    for v in candidates:
        if 0 < abs(v) < 1:           # decimal proportion -> percentage
            expanded.append(v * 100)
        elif 1 <= abs(v) <= 100:     # percentage -> decimal
            expanded.append(v / 100)

    for v in expanded:
        if v == 0 and needle == 0:
            return True
        if v == 0:
            if abs(needle) < 1e-9:
                return True
            continue
        if math.isclose(needle, v, rel_tol=rtol, abs_tol=1e-6):
            return True
        if math.isclose(round(needle), round(v), rel_tol=rtol, abs_tol=1.0):
            return True
    return False

EMPTY_SENTINELS = {None, "empty", "empty result", "none", "null", "no data"}


def ground_claim(
    text: str, resolved: Any, measurements: Optional[list] = None
) -> tuple[bool, str]:
    """
    Check the LLM-declared measurements against `resolved`.

    If the LLM provided `measurements`, use them. If not ( older claims, or
    LLM forgot the field ), fall back to extracting numbers from `text` and
    filtering out label-shaped ones via the heuristic.
    """
    is_empty_resolved = (
        resolved is None
        or (isinstance(resolved, (pd.Series, pd.DataFrame)) and len(resolved) == 0)
        or (isinstance(resolved, str) and resolved.lower() in EMPTY_SENTINELS)
    )
    if is_empty_resolved and not measurements:
        return True, "claim asserts emptiness; resolved value is empty (consistent)"

    if measurements is None:
        # Fallback to the regex extraction + label heuristic.
        raw = _extract_numbers(text)
        needles = [n for n in raw if not _is_label_number(n, text)]
    else:
        # Trust the LLM's declaration.
        needles = []
        for m in measurements:
            try:
                needles.append(float(m))
            except (TypeError, ValueError):
                continue

    if not needles:
        return True, "no measurement numbers to verify"
    haystack = _values_from(resolved)
    if not haystack:
        return (
            False,
            f"text contains measurements {needles} but resolved value has none",
        )
    unmatched = [n for n in needles if not _matches(n, haystack)]
    if unmatched:
        detail = (
            f"measurements {unmatched} not found in resolved value "
            f"of length {len(haystack)} (sample: {haystack[:8]})"
        )
        return False, detail
    return True, f"all {len(needles)} measurement(s) match the resolved value"


# Numbers that look like calendar years, percentages-of-100, or ordinals are *labels*, not measurements. Don't try to ground them against cells.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
_PERCENTILE_RE = re.compile(
    r"\b(?:25|50|75)\s*%|\b(?:25th|50th|75th|90th|95th|99th)\b", re.IGNORECASE
)
_QUARTILE_WORDS = {"quartile", "percentile", "median", "quantile"}


def _is_label_number(num: float, text: str) -> bool:
    """
    Heuristics: a number in claim text that's a label, not a measurement to verify.
    """
    # Calendar years (1900-2099)
    if 1900 <= num <= 2099 and num == int(num):
        if _YEAR_RE.search(text):
            return True
    lowered = text.lower()
    # Numbers next to month names
    for month in _MONTH_NAMES:
        if month in lowered and 1900 <= num <= 2099:
            return True
    # Percentile/quartile labels: "75% of products", "25th percentile", etc.
    if _PERCENTILE_RE.search(text) or any(w in lowered for w in _QUARTILE_WORDS):
        if num in (25, 50, 75, 90, 95, 99):
            return True
    return False

# -------------------------------------------
# Public entry point
# ------------------------------------------------------

def resolve_claims(result, raw_claims):
    out = []
    for c in raw_claims:
        text = str(c.get("text", "")).strip()
        kind = str(c.get("evidence_kind", "")).strip()
        ref = c.get("evidence_ref") or {}
        measurements = c.get("measurements")
        rc = ResolvedClaim(text=text, evidence_kind=kind,
                           evidence_ref=ref if isinstance(ref, dict) else {})
        try:
            rc.resolved_value = resolve(result, kind, rc.evidence_ref)
        except Exception as e:
            rc.resolution_error = f"{type(e).__name__}: {e}"
            out.append(rc); continue

        ground_target = rc.resolved_value
        needles = measurements if measurements is not None else _extract_numbers(text)

        # Multi-measurement broadening:
        # When a claim cites multiple numbers, expose more of the result so all of them have a chance to ground.
        if isinstance(needles, list) and len(needles) > 1:
            if kind == "cell" and isinstance(result, pd.DataFrame):
                # expose the whole row.
                row_key = ref.get("row")
                try:
                    row = _lookup_row(result, row_key)
                    if isinstance(row, pd.Series):
                        ground_target = (rc.resolved_value, row)
                except Exception:
                    pass
            elif kind == "stat" and isinstance(result, (pd.DataFrame, pd.Series)):
                # for "X ranges from A to B" claims with stat=min/max, also expose the entire column.
                col = ref.get("col")
                col_data = None
                if isinstance(result, pd.Series):
                    col_data = result
                elif isinstance(result, pd.DataFrame) and col:
                    actual = _lookup_column(result, col)
                    if actual is not None:
                        col_data = result[actual]
                if col_data is not None:
                    ground_target = (rc.resolved_value, col_data)

        rc.grounded, rc.grounding_detail = ground_claim(
            text, ground_target, measurements=measurements)
        out.append(rc)
    return out


