"""
TableStore - SQLite-backed registry for the loaded CSVs.

Why SQLite for a PoC?
  - Free schema/foreign-key support
  - Disk-resident: 5×500MB CSVs don't blow RAM
  - SQL is often clearer than pandas for multi-table joins
  - Trivially swappable later for Postgres/DuckDB/etc - same API surface

What lives here:
  - ingest_csv(path, name, progress_cb): chunked load CSV -> SQLite table
  - list_tables() / schema() / preview()
  - register_foreign_key(...): add an FK after the user confirms a candidate
  - query_pandas(sql) / exec_pandas(sql): run SQL, return DataFrame
  - to_dataframes(): materialise all tables as a dict[str, DataFrame] for
    pandas-engine code that needs in-memory access

Slug names (`my-file.csv` → `my_file`) are used as SQLite table names; we
sanitise to keep the schema introspectable.
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import pandas as pd

from .data_io import _optimise_dtypes
from .logging_util import get_logger


# --------------------------------
# Helpers
# --------------------------------

_SLUG_RE = re.compile(r"[^A-Za-z0-9_]+")


def slugify_table_name(name: str, existing: Optional[set] = None) -> str:
    """`Customers (2023).csv` -> `customers_2023`. Disambiguates collisions."""
    base = os.path.splitext(os.path.basename(name))[0]
    slug = _SLUG_RE.sub("_", base).strip("_").lower()
    if not slug:
        slug = "table"
    if slug[0].isdigit():
        slug = "t_" + slug
    if existing is None:
        return slug
    candidate = slug
    i = 2
    while candidate in existing:
        candidate = f"{slug}_{i}"
        i += 1
    return candidate


@dataclass
class TableMeta:
    name: str
    source_filename: str
    n_rows: int
    n_cols: int
    columns: list[tuple[str, str]] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list) # post-confirm


# ---------------------
# TableStore
# ------------------------

class TableStore:
    """A wrapper around a SQLite database that holds the user's CSVs."""

    def __init__(self, db_path: Optional[str] = None):
        self._log = get_logger("table_store")
        if db_path is None:
            fd, db_path = tempfile.mkstemp(prefix="data_analyst_", suffix=".sqlite")
            os.close(fd)
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._meta: dict[str, TableMeta] = {}
        self._log.info(f"opened sqlite db at {self.db_path}")

    # ----- ingest -----

    def ingest_csv(
        self,
        path: str,
        table_name: Optional[str] = None,
        chunksize: int = 100_000,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> TableMeta:
        """
        Stream a CSV into SQLite in chunks. `progress_cb(fraction, message)`
        is called periodically so the UI can show progress.
        """
        existing = set(self._meta.keys())
        name = table_name or slugify_table_name(path, existing)
        if name in existing:
            name = slugify_table_name(name, existing)

        size_bytes = os.path.getsize(path)
        if progress_cb:
            progress_cb(0.0, f"opening {os.path.basename(path)} ({size_bytes/1e6:.1f} MB)")

        # Drop any prior table of this name (idempotent on re-upload)
        self._conn.execute(f'DROP TABLE IF EXISTS "{name}"')

        n_rows_total = 0
        first_chunk = True
        # Open the file ourselves so we can report progress by byte position.
        # pd.read_csv accepts a file-like object and we pass chunksize.
        fh = open(path, "rb")
        try:
            reader = pd.read_csv(fh, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                chunk = _optimise_dtypes(chunk)
                chunk.to_sql(name, self._conn,
                             if_exists="replace" if first_chunk else "append",
                             index=False)
                first_chunk = False
                n_rows_total += len(chunk)
                if progress_cb:
                    pos = fh.tell()
                    frac = min(0.99, pos / max(1, size_bytes))
                    progress_cb(frac, f"loaded {n_rows_total:,} rows…")
        finally:
            fh.close()

        # Pull schema back from sqlite for the meta object
        cols = self._conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        col_pairs = [(c[1], c[2]) for c in cols]   # (name, sqlite type)

        meta = TableMeta(
            name=name,
            source_filename=os.path.basename(path),
            n_rows=n_rows_total,
            n_cols=len(col_pairs),
            columns=col_pairs,
        )
        self._meta[name] = meta
        if progress_cb:
            progress_cb(1.0, f"done - {n_rows_total:,} rows × {len(col_pairs)} cols")
        self._log.info(f"ingested {path} -> table '{name}' "
                       f"({n_rows_total:,} rows, {len(col_pairs)} cols)")
        return meta

    # ----- introspection -----

    def list_tables(self) -> list[TableMeta]:
        return list(self._meta.values())

    def schema(self, name: str) -> TableMeta:
        return self._meta[name]

    def preview(self, name: str, n: int = 20) -> pd.DataFrame:
        return pd.read_sql(f'SELECT * FROM "{name}" LIMIT {n}', self._conn)

    def distinct_values(self, table: str, column: str, limit: int = 5000) -> pd.Series:
        """For relationship discovery: distinct values in a column."""
        q = f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT {limit}'
        return pd.read_sql(q, self._conn).iloc[:, 0]

    def column_stats(self, table: str, column: str) -> dict:
        """Lightweight stats - count, distinct count - for FK heuristics."""
        q = (f'SELECT COUNT(*) AS n, COUNT(DISTINCT "{column}") AS n_distinct '
             f'FROM "{table}"')
        row = self._conn.execute(q).fetchone()
        return {"n": row[0], "n_distinct": row[1]}

    # ----- foreign keys (post-user-confirmation) -----
    def register_foreign_key(self, child_table: str, child_col: str,
                             parent_table: str, parent_col: str) -> None:
        # Skip if already registered
        for fk in self._meta[child_table].foreign_keys:
            if (fk["child_col"] == child_col
                    and fk["parent_table"] == parent_table
                    and fk["parent_col"] == parent_col):
                return

        meta = self._meta[child_table]
        cols_def = ", ".join(f'"{c}" {t}' for c, t in meta.columns)
        col_list = ", ".join(f'"{c}"' for c, _ in meta.columns)
        tmp = f"_tmp_{child_table}"

        # Capture the current row count so we can verify the migration.
        original_count = self._conn.execute(
            f'SELECT COUNT(*) FROM "{child_table}"').fetchone()[0]

        # Disable FK enforcement during the migration. Otherwise rows whose parent key is not yet present (or whose dtypes mismatch) will be rejected silently row-by-row, leaving an empty table.
        cur = self._conn.cursor()
        try:
            cur.execute("PRAGMA foreign_keys = OFF")
            cur.execute("BEGIN")

            # Drop any leftover _tmp_ table from a previous failed attempt.
            cur.execute(f'DROP TABLE IF EXISTS "{tmp}"')

            cur.execute(f'ALTER TABLE "{child_table}" RENAME TO "{tmp}"')
            cur.execute(
                f'CREATE TABLE "{child_table}" ({cols_def}, '
                f'FOREIGN KEY("{child_col}") REFERENCES '
                f'"{parent_table}"("{parent_col}"))'
            )
            cur.execute(
                f'INSERT INTO "{child_table}" ({col_list}) '
                f'SELECT {col_list} FROM "{tmp}"'
            )

            # Verify before we commit. If counts don't match, we have a bug
            # and we'd rather raise than silently leave an empty table.
            new_count = cur.execute(
                f'SELECT COUNT(*) FROM "{child_table}"').fetchone()[0]
            if new_count != original_count:
                raise RuntimeError(
                    f"FK registration would drop rows: had {original_count}, "
                    f"got {new_count}. Aborting."
                )

            cur.execute(f'DROP TABLE "{tmp}"')
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            # if the rename happened but the rest didn't, swap back.
            try:
                cur.execute(f'DROP TABLE IF EXISTS "{child_table}"')
                cur.execute(f'ALTER TABLE "{tmp}" RENAME TO "{child_table}"')
                self._conn.commit()
            except Exception:
                pass
            raise
        finally:
            cur.execute("PRAGMA foreign_keys = ON")

        meta.foreign_keys.append({
            "child_col": child_col,
            "parent_table": parent_table,
            "parent_col": parent_col,
        })
        self._log.info(
            f"FK registered: {child_table}.{child_col} -> "
            f"{parent_table}.{parent_col} ({original_count:,} rows preserved)"
        )
    # ----- query interface -----

    def query_pandas(self, sql: str) -> pd.DataFrame:
        """Run a read-only SQL query and return a DataFrame."""
        return pd.read_sql(sql, self._conn)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Materialise all tables for pandas-engine code paths."""
        out: dict[str, pd.DataFrame] = {}
        for name in self._meta:
            out[name] = pd.read_sql(f'SELECT * FROM "{name}"', self._conn)
        return out

    # ----- LLM-friendly schema dump -----

    def schema_for_prompt(self) -> str:
        """Compact schema description suitable for an LLM system message."""
        lines: list[str] = []
        for meta in self._meta.values():
            lines.append(f'TABLE "{meta.name}"  ({meta.n_rows:,} rows, source: {meta.source_filename})')
            for col, dtype in meta.columns:
                lines.append(f'  - "{col}"  {dtype}')
            for fk in meta.foreign_keys:
                lines.append(f'  FK: {meta.name}."{fk["child_col"]}" -> '
                             f'{fk["parent_table"]}."{fk["parent_col"]}"')
            lines.append("")
        return "\n".join(lines).rstrip()

    # ----- cleanup -----

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()