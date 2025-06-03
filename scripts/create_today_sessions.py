import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database import Class, ClassSession
from database.db import SessionLocal
from crud.class_crud import create_class_session
from schemas.class_schema import ClassSessionCreate

def create_today_sessions():
    """Create class sessions for today following the MWF or TTh pattern"""
    db = SessionLocal()
    try:
        # Get all classes
        classes = db.query(Class).all()
        if not classes:
            print("No classes found in the database")
            return
        
        # Get today's date and day of week
        today = datetime.now().date()
        day_of_week = today.weekday()  # 0=Monday, 1=Tuesday, etc.
        
        # Determine if today is a class day (MWF or TTh)
        is_mwf = day_of_week in [0, 2, 4]  # Monday, Wednesday, Friday
        is_tth = day_of_week in [1, 3]     # Tuesday, Thursday
        
        if not (is_mwf or is_tth):
            print(f"Today is {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]}, which is not a regular class day")
            choice = input("Do you want to create sessions anyway? (y/n): ")
            if choice.lower() != 'y':
                return
        
        # Create sessions for each class
        sessions_created = 0
        for i, class_obj in enumerate(classes):
            # Check if a session already exists for today
            existing_session = db.query(ClassSession).filter(
                ClassSession.class_id == class_obj.id,
                ClassSession.session_date == today
            ).first()
            
            if existing_session:
                print(f"Session for {class_obj.name} already exists for today")
                continue
            
            # Each class has a different start time based on the seed data pattern
            class_hour = 9 + i % 3  # 9AM, 10AM, 11AM (following seed pattern)
            
            # Create session
            session = ClassSessionCreate(
                class_id=class_obj.id,
                session_date=today,
                start_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=class_hour),
                end_time=datetime.combine(today, datetime.min.time()) + timedelta(hours=class_hour+1, minutes=30),
                notes=f"{['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]} session for {class_obj.name}"
            )
            
            # Store in database
            created_session = create_class_session(db, session)
            sessions_created += 1
            
            print(f"Created session for {class_obj.name} at {created_session.start_time.strftime('%H:%M')} - {created_session.end_time.strftime('%H:%M')}")
        
        print(f"\nTotal sessions created: {sessions_created}")
        
    except Exception as e:
        print(f"Error creating sessions: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    create_today_sessions()