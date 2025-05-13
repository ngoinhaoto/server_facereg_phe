import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database import User, Class, ClassSession, Attendance, FaceEmbedding, AttendanceStatus
from database.db import SessionLocal
from security.password import get_password_hash

def create_sample_data():
    db = SessionLocal()
    try:
        # Create users with different roles
        teacher = User(
            username="prof_smith",
            email="smith@university.edu",
            hashed_password=get_password_hash("teacher123"),
            role="teacher"
        )
        
        students = [
            User(
                username=f"student{i}",
                email=f"student{i}@university.edu",
                hashed_password=get_password_hash(f"student{i}"),
                role="student"
            )
            for i in range(1, 6)  # Create 5 students
        ]
        
        admin = User(
            username="admin",
            email="admin@university.edu",
            hashed_password=get_password_hash("admin123"),
            role="admin"
        )
        
        # Add users to session
        db.add(teacher)
        for student in students:
            db.add(student)
        db.add(admin)
        db.flush()  # Flush to get IDs
        
        # Create classes
        classes = [
            Class(
                class_code=f"CS{i}01",
                name=f"Computer Science {i}01",
                description=f"Introduction to Computer Science topic {i}",
                semester="Fall",
                academic_year="2025",
                teacher_id=teacher.id,
                start_time=datetime.now().replace(hour=10, minute=0),
                end_time=datetime.now().replace(hour=11, minute=30),
                location=f"Building A, Room {i}01"
            )
            for i in range(1, 3)  # Create 2 classes
        ]
        
        for class_obj in classes:
            db.add(class_obj)
        db.flush()
        #
        # Register students to classes
        for student in students:
            for class_obj in classes:
                student.classes.append(class_obj)
        
        # Create class sessions
        today = datetime.now().date()
        sessions = []
        
        for i, class_obj in enumerate(classes):
            for day_offset in range(5):  # Create 5 sessions per class
                session_date = today - timedelta(days=day_offset)
                session = ClassSession(
                    class_id=class_obj.id,
                    session_date=session_date,
                    start_time=datetime.combine(session_date, datetime.min.time()) + timedelta(hours=10),
                    end_time=datetime.combine(session_date, datetime.min.time()) + timedelta(hours=11, minutes=30),
                    notes=f"Session {day_offset+1} for {class_obj.name}"
                )
                sessions.append(session)
                db.add(session)
        db.flush()
        
        # Create attendance records
        for session in sessions:
            for i, student in enumerate(students):
                # Vary attendance status for demonstration
                if i % 4 == 0:
                    status = AttendanceStatus.ABSENT.value
                    check_in_time = None
                    late_minutes = 0
                elif i % 3 == 0:
                    status = AttendanceStatus.LATE.value
                    check_in_time = session.start_time + timedelta(minutes=15)
                    late_minutes = 15
                else:
                    status = AttendanceStatus.PRESENT.value
                    check_in_time = session.start_time - timedelta(minutes=5)
                    late_minutes = 0
                
                attendance = Attendance(
                    student_id=student.id,
                    session_id=session.id,
                    status=status,
                    check_in_time=check_in_time,
                    late_minutes=late_minutes
                )
                db.add(attendance)
        
        # Create dummy face embeddings (normally these would be encrypted vectors)
        for student in students:
            # Create a dummy binary embedding for demonstration
            dummy_embedding = bytes([i % 256 for i in range(128)])
            embedding = FaceEmbedding(
                user_id=student.id,
                encrypted_embedding=dummy_embedding,
                confidence_score=0.95,
                device_id="test_device_001"
            )
            db.add(embedding)
        
        db.commit()
        print("Sample data created successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating sample data: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # Create a simple password hashing module first
    if not os.path.exists("/Users/ritherthemuncher/Desktop/server_facereg/security"):
        os.makedirs("/Users/ritherthemuncher/Desktop/server_facereg/security")
    
    password_module = """
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
"""
    
    with open("/Users/ritherthemuncher/Desktop/server_facereg/security/password.py", "w") as f:
        f.write(password_module.strip())
    
    # Install required password hashing library if not already installed
    try:
        import passlib
    except ImportError:
        print("Installing passlib...")
        os.system("pip install passlib[bcrypt]")
    
    # Run the data creation
    create_sample_data()