"""Seed all KIET branches and sections.

Usage:
    export ATTEND_DB_DSN="postgresql://attend:attend_dev_pw@localhost:5432/attendance"
    python -m registration.seed

Creates:
  - 16 branches
  - 2 sections (A, B) × 4 years per branch  →  128 sections total

If ATTEND_DB_DSN is not set (in-memory mode), prints a fixed demo UUID.
"""

import os

# ── College hierarchy ─────────────────────────────────────────────────────-

# (branch_code, branch_name)
BRANCHES = [
    # CS / IT cluster
    ("CSIT",     "Computer Science & Information Technology"),
    ("CSE",      "Computer Science & Engineering"),
    ("CS",       "Computer Science"),
    ("CSE-AIML", "CSE (AI & Machine Learning)"),
    ("CSE-AI",   "CSE (Artificial Intelligence)"),
    ("IT",       "Information Technology"),
    ("CSE-DS",   "CSE (Data Science)"),
    ("CSE-CS",   "CSE (Cyber Security)"),
    # Electronics cluster
    ("ECE",      "Electronics & Communication Engineering"),
    ("EEE",      "Electrical & Electronics Engineering"),
    ("ELCE",     "Electronics & Computer Engineering"),
    ("ECE-VLSI", "ECE (VLSI Design)"),
    # Mechanical cluster
    ("ME",       "Mechanical Engineering"),
    ("AM-AI",    "Applied Mechanics & AI"),
    # Service / other
    ("ASH",      "Applied Science & Humanities"),
    ("BPHARM",   "Bachelor of Pharmacy"),
]

SECTIONS = ["A", "B","C","D"]   # add C, D manually if a branch grows beyond 2 sections
YEARS    = [1, 2, 3, 4]

# fixed UUID printed when running in-memory (no DB)
DEMO_SECTION_ID = "11111111-1111-1111-1111-111111111111"


def main():
    dsn = os.getenv("ATTEND_DB_DSN")
    if not dsn:
        print("No ATTEND_DB_DSN set — in-memory mode.")
        print("Use any Section ID on the page (foreign keys not enforced):")
        print("  ", DEMO_SECTION_ID)
        return

    import psycopg

    conn = psycopg.connect(dsn, autocommit=True)
    cur  = conn.cursor()

    # ── branches ─────────────────────────────────────────────────────────
    branch_ids = {}
    for code, name in BRANCHES:
        cur.execute(
            "INSERT INTO branches (code, name) VALUES (%s, %s)"
            " ON CONFLICT (code) DO NOTHING",
            (code, name),
        )
        cur.execute("SELECT id FROM branches WHERE code = %s", (code,))
        branch_ids[code] = cur.fetchone()[0]

    # ── sections ─────────────────────────────────────────────────────────
    section_count = 0
    for branch_code, _ in BRANCHES:
        bid = branch_ids[branch_code]
        for year in YEARS:
            for sec in SECTIONS:
                cur.execute(
                    "INSERT INTO sections (branch_id, name, year) VALUES (%s, %s, %s)"
                    " ON CONFLICT (branch_id, name, year) DO NOTHING",
                    (bid, sec, year),
                )
                section_count += 1

    conn.close()

    print(f"Seeded {len(BRANCHES)} branches, {section_count} sections.")
    print()
    print("Students open  http://<server>:8000/  and select their branch/year/section.")


if __name__ == "__main__":
    main()
