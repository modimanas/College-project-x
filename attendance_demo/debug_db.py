#!/usr/bin/env python
"""Debug script to check database embeddings"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_demo import SessionLocal
from sqlalchemy import text

session = SessionLocal()

print("=" * 60)
print("DATABASE DEBUG INFO")
print("=" * 60)

# Check total students
total = session.execute(text("SELECT COUNT(*) FROM students")).scalar()
print(f"Total students in DB: {total}")

# Check students with embeddings
with_emb = session.execute(text("""
    SELECT COUNT(*) FROM students 
    WHERE embedding IS NOT NULL
""")).scalar()
print(f"Students WITH embeddings: {with_emb}")

# Check students without embeddings
without_emb = session.execute(text("""
    SELECT COUNT(*) FROM students 
    WHERE embedding IS NULL
""")).scalar()
print(f"Students WITHOUT embeddings: {without_emb}")

print("\n" + "=" * 60)
print("FIRST 5 STUDENTS:")
print("=" * 60)

rows = session.execute(text("""
    SELECT roll_no, student_name, embedding
    FROM students 
    LIMIT 5
""")).fetchall()

for roll_no, name, emb in rows:
    emb_status = "✅ HAS" if emb is not None else "❌ NULL"
    print(f"{roll_no:20} | {name:30} | {emb_status}")

print("\n" + "=" * 60)

if without_emb > 0:
    print(f"⚠️  {without_emb} students missing embeddings!")
    print("❌ REGISTRATION NEEDED - run: python registration/register.py")
else:
    print("✅ All students have embeddings!")

session.close()
