from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, ForeignKey, Boolean, Float, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid
import enum

Base = declarative_base()

# Enum for attendance status
class AttendanceStatus(enum.Enum):
    PRESENT = "present"
    LATE = "late" 
    ABSENT = "absent"
    
# Many-to-many relationship table between students and classes
student_class_association = Table(
    'student_class_association',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('class_id', Integer, ForeignKey('classes.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="student")
    full_name = Column(String, nullable=True)  # Add this line

    # Add these new fields
    student_id = Column(String, unique=True, index=True, nullable=True)
    staff_id = Column(String, unique=True, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())    
    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete")
    # New relationships
    classes = relationship("Class", secondary=student_class_association, back_populates="students")
    attendances = relationship("Attendance", back_populates="student")
    taught_classes = relationship("Class", back_populates="teacher")

class Class(Base):
    __tablename__ = "classes"
    
    id = Column(Integer, primary_key=True, index=True)
    class_code = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    semester = Column(String)
    academic_year = Column(String)
    teacher_id = Column(Integer, ForeignKey("users.id"))
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    location = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())  # Add this line
    
    # Relationships
    students = relationship("User", secondary=student_class_association, back_populates="classes")
    teacher = relationship("User", back_populates="taught_classes", foreign_keys=[teacher_id])
    sessions = relationship("ClassSession", back_populates="class_obj", cascade="all, delete")

class ClassSession(Base):
    __tablename__ = "class_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"))
    session_date = Column(DateTime(timezone=True), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    notes = Column(String)
    
    # Relationships
    class_obj = relationship("Class", back_populates="sessions")
    attendances = relationship("Attendance", back_populates="session", cascade="all, delete")

class Attendance(Base):
    __tablename__ = "attendances"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    session_id = Column(Integer, ForeignKey("class_sessions.id", ondelete="CASCADE"))
    status = Column(String, default=AttendanceStatus.ABSENT.value)
    check_in_time = Column(DateTime(timezone=True))
    late_minutes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    student = relationship("User", back_populates="attendances")
    session = relationship("ClassSession", back_populates="attendances")
    
class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    
    # Store the encrypted face embedding vector
    encrypted_embedding = Column(LargeBinary, nullable=False)
    
    # Optional metadata
    confidence_score = Column(Float)
    device_id = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="face_embeddings")
    
class AuthLog(Base):
    __tablename__ = "auth_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    device_id = Column(String, index=True)
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String)

