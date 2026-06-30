from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    roll_no = Column(
        String(30),
        unique=True,
        nullable=False,
        index=True
    )

    student_name = Column(
        String(255),
        nullable=False
    )

    image_path = Column(
        Text,
        nullable=False
    )

    embedding = Column(
        ARRAY(Float),
        nullable=False
    )

    model_name = Column(
        String(50),
        nullable=False,
        default="antelopev2"
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    def __repr__(self):
        return (
            f"<Student("
            f"roll_no='{self.roll_no}', "
            f"name='{self.student_name}'"
            f")>"
        )