#!/usr/bin/env python
"""Show UNIQUE present students from attendance log"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_demo import SessionLocal
from sqlalchemy import text

session = SessionLocal()

print("\n" + "=" * 80)
print(" " * 20 + "PRESENT STUDENTS - UNIQUE ONLY")
print("=" * 80)

# Get UNIQUE students by roll_no (group by and take min time = first marking)
rows = session.execute(text("""
    SELECT DISTINCT ON (roll_no)
        roll_no,
        student_name,
        marked_at
    FROM attendance_log
    ORDER BY roll_no, marked_at ASC
""")).fetchall()

print(f"\n✅ Total UNIQUE Present: {len(rows)} students\n")

for i, (roll_no, name, marked_at) in enumerate(rows, 1):
    time_str = marked_at.strftime('%H:%M:%S') if marked_at else 'N/A'
    print(f"{i:3}. {name:40} | Roll: {roll_no:20} | Marked @ {time_str}")

print("\n" + "=" * 80)
print(f"Session Summary: {len(rows)}/{76} students marked present")
print("=" * 80 + "\n")

session.close()

