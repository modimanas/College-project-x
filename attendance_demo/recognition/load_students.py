import numpy as np
from sqlalchemy import text

def load_students(session):
    students = []

    rows = session.execute(
        text("""
        SELECT
            roll_no,
            student_name,
            embedding
        FROM students
        """)
    )

    for row in rows:
        # Check if the database row contains data
        if row.embedding is None:
            continue
            
        # Safe fallback parsing: If it arrives as a string format "[0.1,-0.2...]"
        if isinstance(row.embedding, str):
            # Clean brackets and split string into numeric list array
            clean_str = row.embedding.strip('[]')
            embedding_data = [float(x) for x in clean_str.split(',') if x.strip()]
        else:
            # If pgvector driver automatically passes it down as a Python list/array
            embedding_data = row.embedding

        students.append({
            "roll_no": row.roll_no,
            "name": row.student_name,
            "embedding": np.array(
                embedding_data,
                dtype=np.float32
            )
        })

    return students