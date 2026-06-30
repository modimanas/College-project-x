# registration/register.py
import os
import sys
import cv2
import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database_demo import SessionLocal
from facex.embedder import default_embedder


def parse_filename(filename):
    """
    Parses 'NAME_SURNAME_ROLLNUMBER.jpg' into clean components.
    Example: 'AAKIF_MUSHTAQ_SIDDIQUI_697442420116.jpg'
    Returns: name = "AAKIF MUSHTAQ SIDDIQUI", roll_no = "697442420116"
    """
    # Remove file extension (.jpg, .png, etc.)
    base_name, _ = os.path.splitext(filename)
    
    # Split the string by underscores
    parts = base_name.split('_')
    
    if len(parts) < 2:
        print(f"⚠️ Skipping invalid filename format: {filename}")
        return None, None
        
    # The last block is the roll number, everything before it forms the student's name
    roll_no = parts[-1]
    student_name = " ".join(parts[:-1])
    
    return student_name, roll_no


_default_uploads = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")

def register_students_from_folder(folder_path=_default_uploads):
    """
    Scans the folder, generates embeddings, and saves records into pgvector columns.
    """
    session = SessionLocal()
    
    if not os.path.exists(folder_path):
        print(f"❌ Target directory '{folder_path}' does not exist.")
        return

    print("📁 Processing student registration profiles...")
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Ignore system hidden files or log records
            
        student_name, roll_no = parse_filename(filename)
        if not student_name or not roll_no:
            continue
            
        img_path = os.path.join(folder_path, filename)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"❌ Failed to load image file: {filename}")
            continue

        # Extract face features using our unified embedder architecture
        faces = default_embedder.detect(frame)
        
        if len(faces) == 0:
            print(f"⚠️ No faces detected in image profile for {student_name}. Skipping.")
            continue
        elif len(faces) > 1:
            print(f"⚠️ Multiple faces detected in reference image for {student_name}! Use single-face IDs.")
            continue
            
        # Extract the processed 512-d embedding container
        face_data = default_embedder.embed_face(faces[0], frame)
        embedding_vector = face_data.vector  # This is a NumPy array of shape (512,)
        
        # Convert the numpy float32 array into a clean standard Python list or string format
        # pgvector handles Python lists or formatted strings like '[0.1, 0.2, ...]' directly
        vector_payload = embedding_vector.tolist()

        try:
            # Check if student record already exists to prevent duplication crashes
            check_exist = session.execute(
                text("SELECT 1 FROM students WHERE roll_no = :roll"), 
                {"roll": roll_no}
            ).fetchone()
            
            if check_exist:
                # Update existing record embedding
                session.execute(
                    text("""
                    UPDATE students 
                    SET student_name = :name, embedding = :emb 
                    WHERE roll_no = :roll
                    """),
                    {"name": student_name, "emb": vector_payload, "roll": roll_no}
                )
                print(f"🔄 Updated reference embedding profile for: {student_name}")
            else:
                # Insert a brand new student row
                session.execute(
                    text("""
                    INSERT INTO students (roll_no, student_name, embedding) 
                    VALUES (:roll, :name, :emb)
                    """),
                    {"roll": roll_no, "name": student_name, "emb": vector_payload}
                )
                print(f"✅ Successfully registered student: {student_name} ({roll_no})")
                
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error inserting record database for {student_name}: {str(e)}")

    session.close()
    print("🏁 Student database registration processing completed!")


if __name__ == "__main__":
    # You can run this file directly to populate your PostgreSQL database before hitting demo.py
    register_students_from_folder()