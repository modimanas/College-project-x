#!/usr/bin/env python
"""Clear attendance log and show unique students only"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_demo import SessionLocal
from sqlalchemy import text

session = SessionLocal()

print("\n" + "=" * 70)
print("CLEARING OLD ATTENDANCE RECORDS")
print("=" * 70)

# Clear attendance log
try:
    count = session.execute(text("SELECT COUNT(*) FROM attendance_log")).scalar()
    print(f"\nOld records: {count}")
    
    session.execute(text("DELETE FROM attendance_log"))
    session.commit()
    print(f"✅ Cleared {count} records")
except Exception as e:
    print(f"❌ Error: {e}")
    session.rollback()

print("\n" + "=" * 70)
print("Ready for fresh attendance session!")
print("=" * 70 + "\n")

session.close()
