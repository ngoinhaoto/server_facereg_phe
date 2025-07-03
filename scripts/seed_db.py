import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database import User, Class, ClassSession, Attendance, FaceEmbedding, AttendanceStatus
from database.db import SessionLocal
from security.password import get_password_hash
import random
import numpy as np
import pickle

def create_sample_data():
    db = SessionLocal()
    try:
        # Create users with different roles and appropriate IDs
        teacher = User(
            username="prof_smith",
            email="smith@university.edu",
            full_name="John Pork Smith",
            hashed_password=get_password_hash("teacher123"),
            role="teacher",
            staff_id="STAFF-001"
        )
        
        # Create students with student IDs
        students = [
            User(
                username=f"student{i}",
                email=f"student{i}@university.edu",
                full_name=f"Student {i} Lastname",
                hashed_password=get_password_hash(f"student{i}"),
                role="student",
                student_id=f"STU-{1000+i}"
            )
            for i in range(1, 11)  # Create 10 students for more varied data
        ]
        
        admin = User(
            username="admin",
            email="admin@university.edu",
            full_name="Admin User",
            hashed_password=get_password_hash("admin123"),
            role="admin",
            staff_id="ADMIN-001"
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
            for i in range(1, 4)  # Create 3 classes
        ]
        
        for class_obj in classes:
            db.add(class_obj)
        db.flush()
        
        # Register students to classes
        # Each student is registered to a random subset of classes
        for student in students:
            # Register to 1 or 2 classes randomly
            student_classes = random.sample(classes, random.randint(1, 2))
            for class_obj in student_classes:
                student.classes.append(class_obj)
        
        # Create class sessions with a structured schedule
        # Each class has sessions on specific days of the week
        today = datetime.now().date()
        # Go back to the most recent Monday
        while today.weekday() != 0:  # 0 is Monday
            today = today - timedelta(days=1)
        
        # Create 3 weeks of sessions
        sessions = []
        days_of_week = ['Monday', 'Wednesday', 'Friday']  # MWF schedule
        
        for class_idx, class_obj in enumerate(classes):
            # Each class has a different start time
            class_hour = 9 + class_idx  # 9AM, 10AM, 11AM
            
            for week in range(3):  # 3 weeks of history
                for day_idx, day_name in enumerate(days_of_week):
                    # Calculate the date for this session
                    session_date = today + timedelta(days=day_idx*2) - timedelta(weeks=week)
                    
                    # Create the session
                    session = ClassSession(
                        class_id=class_obj.id,
                        session_date=session_date,
                        start_time=datetime.combine(session_date, datetime.min.time()) + timedelta(hours=class_hour),
                        end_time=datetime.combine(session_date, datetime.min.time()) + timedelta(hours=class_hour+1, minutes=30),
                        notes=f"{day_name} {week+1} for {class_obj.name}"
                    )
                    sessions.append(session)
                    db.add(session)
        
        db.flush()
        
        # Create attendance records with realistic patterns
        # Some students are consistently on time, some are late, some miss classes
        
        # Assign attendance patterns to students
        attendance_patterns = {
            students[0].id: {"always_present": True, "usually_early": True},     # Always early
            students[1].id: {"always_present": True, "usually_on_time": True},   # Always on time
            students[2].id: {"always_present": True, "usually_late": True},      # Always present but late
            students[3].id: {"sometimes_absent": True, "absence_rate": 0.3},     # Sometimes absent (30%)
            students[4].id: {"always_present": True, "random_timing": True},     # Random arrival times
            students[5].id: {"rarely_absent": True, "absence_rate": 0.1},        # Rarely absent (10%)
            students[6].id: {"often_absent": True, "absence_rate": 0.5},         # Often absent (50%)
            students[7].id: {"always_late": True, "late_minutes": [5, 15]},      # Always late (5-15 min)
            students[8].id: {"extremely_late": True, "late_minutes": [20, 45]},  # Extremely late sometimes
            students[9].id: {"perfect": True},                                   # Perfect attendance, always early
        }
        
        # Record all attendance
        for session in sessions:
            # Get all students registered for this class
            class_students = db.query(User).join(User.classes).filter(
                Class.id == session.class_id, 
                User.role == "student"
            ).all()
            
            for student in class_students:
                pattern = attendance_patterns.get(student.id, {"random": True})
                
                # Determine attendance status based on pattern
                if pattern.get("always_present", False) or pattern.get("perfect", False):
                    status = AttendanceStatus.PRESENT.value
                elif pattern.get("sometimes_absent", False) and random.random() < pattern.get("absence_rate", 0.3):
                    status = AttendanceStatus.ABSENT.value
                elif pattern.get("rarely_absent", False) and random.random() < pattern.get("absence_rate", 0.1):
                    status = AttendanceStatus.ABSENT.value
                elif pattern.get("often_absent", False) and random.random() < pattern.get("absence_rate", 0.5):
                    status = AttendanceStatus.ABSENT.value
                elif pattern.get("random", False) and random.random() < 0.2:  # 20% chance of absence for random pattern
                    status = AttendanceStatus.ABSENT.value
                else:
                    status = AttendanceStatus.PRESENT.value
                
                # Determine check-in time and late status
                if status == AttendanceStatus.ABSENT.value:
                    check_in_time = None
                    late_minutes = 0
                else:
                    # Determine arrival time based on pattern
                    if pattern.get("perfect", False):
                        # Always 10 minutes early
                        check_in_time = session.start_time - timedelta(minutes=10)
                        late_minutes = 0
                    elif pattern.get("usually_early", False):
                        # 5-10 minutes early
                        early_minutes = random.randint(5, 10)
                        check_in_time = session.start_time - timedelta(minutes=early_minutes)
                        late_minutes = 0
                    elif pattern.get("usually_on_time", False):
                        # 0-5 minutes early
                        early_minutes = random.randint(0, 5)
                        check_in_time = session.start_time - timedelta(minutes=early_minutes)
                        late_minutes = 0
                    elif pattern.get("usually_late", False) or pattern.get("always_late", False):
                        # 5-15 minutes late by default
                        min_late, max_late = pattern.get("late_minutes", [5, 15])
                        late_minutes = random.randint(min_late, max_late)
                        check_in_time = session.start_time + timedelta(minutes=late_minutes)
                        status = AttendanceStatus.LATE.value
                    elif pattern.get("extremely_late", False) and random.random() < 0.7:  # 70% chance of being very late
                        late_minutes = random.randint(20, 45)
                        check_in_time = session.start_time + timedelta(minutes=late_minutes)
                        status = AttendanceStatus.LATE.value
                    elif pattern.get("random_timing", False):
                        # Completely random: -15 to +30 minutes from start time
                        minutes_diff = random.randint(-15, 30)
                        check_in_time = session.start_time + timedelta(minutes=minutes_diff)
                        if minutes_diff > 0:
                            late_minutes = minutes_diff
                            status = AttendanceStatus.LATE.value
                        else:
                            late_minutes = 0
                    else:
                        # Default: -5 to +10 minutes
                        minutes_diff = random.randint(-5, 10)
                        check_in_time = session.start_time + timedelta(minutes=minutes_diff)
                        if minutes_diff > 0:
                            late_minutes = minutes_diff
                            status = AttendanceStatus.LATE.value
                        else:
                            late_minutes = 0
                
                attendance = Attendance(
                    student_id=student.id,
                    session_id=session.id,
                    status=status,
                    check_in_time=check_in_time,
                    late_minutes=late_minutes
                )
                db.add(attendance)
        
        db.commit()
        print("Sample data created successfully (without face embeddings)!")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating sample data: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # Password hashing module setup code unchanged
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