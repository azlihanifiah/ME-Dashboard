from __future__ import annotations

import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path(__file__).resolve().parents[1] / "data" / "regdata.db"
    print("db_path:", db_path)
    print("exists:", db_path.exists())
    if not db_path.exists():
        return

    print("size:", db_path.stat().st_size)

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        print("tables:", tables)
        for t in tables:
            cur.execute(f"PRAGMA table_info('{t}')")
            cols = cur.fetchall()
            print("-", t, [(r[1], r[2]) for r in cols])

        if "RegData" in tables:
            try:
                cur.execute("SELECT userID, QRID, name, level FROM RegData LIMIT 10")
                rows = cur.fetchall() or []
                print("sample_rows(RegData):")
                for r in rows:
                    print("  ", r)
            except Exception as e:
                print("sample_rows(RegData) error:", e)
    finally:
        con.close()


if __name__ == "__main__":
    main()
