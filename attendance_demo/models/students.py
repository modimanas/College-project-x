# models/student.py
from sqlalchemy import Column, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector  # <-- The pgvector wrapper

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'

    # BigInteger handles large numeric values safely (like 12-digit roll numbers)
    roll_no = Column(String(50), primary_key=True)
    student_name = Column(String(150), nullable=False)
    
    # Strictly locks the column shape to a 512-dimensional vector space
    embedding = Column(Vector(512), nullable=False)

    def __repr__(self):
        return f"<Student(name='{self.student_name}', roll_no='{self.roll_no}')>"