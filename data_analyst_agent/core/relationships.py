"""
Relationship discovery (deterministic, no LLM).

Given a TableStore, find candidate (parent_table.parent_col -> child_table.child_col)
relationships and rank them by confidence. The LLM can later *describe* these in
natural language, but it doesn't *find* them - that's a set-overlap problem with
exact answers.

What we look for, in priority order:

  1. Same-name columns where dtypes are compatible.
  2. Columns where one side has unique values (PK candidate) and the other side's
     values are largely contained in it (FK candidate).
  3. High value-overlap between distinct values, even without a name match.

We sample distinct values up to a cap so that comparing two large tables is
cheap. With a 5000-distinct-value cap, comparing a table of 5M rows to another
of 5M rows is two trivial DISTINCT scans.

Output is a ranked list of `RelationshipCandidate`. The user must confirm
candidates before they become real foreign keys in the store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .logging_util import get_logger
from .table_store import TableStore


@dataclass
class RelationshipCandidate:
    parent_table: str
    parent_col: str
    child_table: str
    child_col: str
    kind: str                          # "shared_name_pk_fk" | "shared_name_overlap" | "value_overlap"
    confidence: float                  # 0..1
    evidence: dict = field(default_factory=dict)

    def label(self) -> str:
        return (f'{self.child_table}."{self.child_col}" → '
                f'{self.parent_table}."{self.parent_col}"')


# --------------------------------
# Type compatibility 
# ---------------------------------

_NUMERIC_TYPES = {"INTEGER", "REAL", "NUMERIC"}
_TEXT_TYPES = {"TEXT", "VARCHAR", "CHAR", ""}


def _types_compatible(a: str, b: str) -> bool:
    a, b = a.upper(), b.upper()
    if a == b:
        return True
    if a in _NUMERIC_TYPES and b in _NUMERIC_TYPES:
        return True
    if a in _TEXT_TYPES and b in _TEXT_TYPES:
        return True
    return False


# -------------------------------
# Discovery
# -------------------------------

def discover_relationships(
    store: TableStore,
    *,
    distinct_cap: int = 5000,
    min_overlap_for_value_match: float = 0.7,
    max_candidates: int = 50,
) -> list[RelationshipCandidate]:
    """
    Candidate relationships across all tables in the store.
    Steps:
      For each (table_a.col_a, table_b.col_b) pair where dtypes are compatible:
        - Pull up to `distinct_cap` distinct values from each side
        - Compute containment: |a ∩ b| / |a|  and  |a ∩ b| / |b|
        - Decide PK direction: whichever side has higher uniqueness fraction
        - Score by: name match (boost), containment, PK clarity
    """
    log = get_logger("relationships")
    tables = store.list_tables()
    if len(tables) < 2:
        return []

    candidates: list[RelationshipCandidate] = []

    # Pre-compute column-level info: distinct values, uniqueness ratio
    col_info: dict[tuple[str, str], dict] = {}
    for tm in tables:
        for col, dtype in tm.columns:
            stats = store.column_stats(tm.name, col)
            uniq_ratio = stats["n_distinct"] / max(1, stats["n"])
            col_info[(tm.name, col)] = {
                "dtype": dtype,
                "n": stats["n"],
                "n_distinct": stats["n_distinct"],
                "uniq_ratio": uniq_ratio,
                "distinct": None,  # lazy
            }

    def _distinct(table: str, col: str) -> set:
        info = col_info[(table, col)]
        if info["distinct"] is None:
            vals = store.distinct_values(table, col, limit=distinct_cap)
            info["distinct"] = set(vals.dropna().tolist())
        return info["distinct"]

    # All ordered pairs across distinct tables
    for i, ta in enumerate(tables):
        for tb in tables[i + 1:]:
            for col_a, dtype_a in ta.columns:
                for col_b, dtype_b in tb.columns:
                    if not _types_compatible(dtype_a, dtype_b):
                        continue

                    info_a = col_info[(ta.name, col_a)]
                    info_b = col_info[(tb.name, col_b)]
                    name_match = col_a.lower() == col_b.lower()
                    plausible_id = (
                        col_a.lower().endswith("id") or col_b.lower().endswith("id")
                        or col_a.lower() == "id" or col_b.lower() == "id"
                    )
                    # Skip pairs that are unlikely to be relationships AT ALL.
                    # (Otherwise N×M of "name" vs "comment" pairs explodes.)
                    if not (name_match or plausible_id):
                        continue

                    # Materialise distinct sets only now (potentially expensive).
                    set_a = _distinct(ta.name, col_a)
                    set_b = _distinct(tb.name, col_b)
                    if not set_a or not set_b:
                        continue
                    inter = set_a & set_b
                    if not inter:
                        continue
                    cont_a_in_b = len(inter) / len(set_a)
                    cont_b_in_a = len(inter) / len(set_b)

                    # Decide direction. The side with higher uniqueness ratio
                    # is the parent (PK candidate); the other points at it.
                    if info_a["uniq_ratio"] >= info_b["uniq_ratio"]:
                        parent_t, parent_c = ta.name, col_a
                        child_t, child_c = tb.name, col_b
                        cont_child_in_parent = cont_b_in_a
                        parent_uniq = info_a["uniq_ratio"]
                    else:
                        parent_t, parent_c = tb.name, col_b
                        child_t, child_c = ta.name, col_a
                        cont_child_in_parent = cont_a_in_b
                        parent_uniq = info_b["uniq_ratio"]

                    # Confidence: combine
                    #   - parent_uniq (1.0 if PK)
                    #   - containment of child values in parent
                    #   - +0.1 boost for name match
                    confidence = (
                        0.5 * parent_uniq +
                        0.5 * cont_child_in_parent +
                        (0.1 if name_match else 0.0)
                    )
                    confidence = min(1.0, confidence)

                    if cont_child_in_parent < min_overlap_for_value_match:
                        continue

                    if name_match and parent_uniq >= 0.99:
                        kind = "shared_name_pk_fk"
                    elif name_match:
                        kind = "shared_name_overlap"
                    elif parent_uniq >= 0.99:
                        kind = "pk_fk_no_name"
                    else:
                        kind = "value_overlap"

                    candidates.append(RelationshipCandidate(
                        parent_table=parent_t,
                        parent_col=parent_c,
                        child_table=child_t,
                        child_col=child_c,
                        kind=kind,
                        confidence=round(confidence, 3),
                        evidence={
                            "name_match": name_match,
                            "parent_unique_ratio": round(parent_uniq, 3),
                            "child_in_parent_overlap": round(cont_child_in_parent, 3),
                            "parent_in_child_overlap": round(
                                cont_a_in_b if parent_t == ta.name else cont_b_in_a, 3),
                            "distinct_sample_capped_at": distinct_cap,
                        },
                    ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    log.info(f"discovered {len(candidates)} candidate relationships")
    return candidates[:max_candidates]