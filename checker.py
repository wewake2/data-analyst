import sqlite3, glob, os, tempfile

candidates = sorted(
    glob.glob(os.path.join(tempfile.gettempdir(), "data_analyst_*.sqlite")),
    key=os.path.getmtime, reverse=True,
)
print("Found:", candidates[:3])
if not candidates:
    raise SystemExit("no db found - make sure the app is running")
db = candidates[0]
con = sqlite3.connect(db)

print("\nTables in db:")
for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    print(f"  {name}: {n} rows")

print("\nReviews rating distribution:")
print(con.execute("SELECT rating, COUNT(*) FROM reviews GROUP BY rating ORDER BY rating").fetchall())

print("\nThe agent's exact query against THIS db:")
q = """SELECT AVG(o.total_amount) AS average_order_value
FROM reviews r INNER JOIN orders o ON r.user_id = o.user_id
WHERE r.rating = 1"""
print(con.execute(q).fetchone())

# Foreign keys currently registered
print("\nForeign keys registered:")
for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    fks = list(con.execute(f'PRAGMA foreign_key_list("{name}")'))
    if fks:
        print(f"  {name}: {fks}")