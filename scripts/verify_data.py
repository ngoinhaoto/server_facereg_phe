import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import SessionLocal
from models.database import User, Class, ClassSession, Attendance

def verify_data():
    db = SessionLocal()
    try:
        # Check users
        users = db.query(User).all()
        print(f"Total users: {len(users)}")
        print(f"Teachers: {len([u for u in users if u.role == 'teacher'])}")
        print(f"Students: {len([u for u in users if u.role == 'student'])}")
        print(f"Admins: {len([u for u in users if u.role == 'admin'])}")
        
        # Check classes
        classes = db.query(Class).all()
        print(f"\nTotal classes: {len(classes)}")
        for class_obj in classes:
            print(f"Class: {class_obj.name}, Students: {len(class_obj.students)}")
        
        # Check sessions
        sessions = db.query(ClassSession).all()
        print(f"\nTotal class sessions: {len(sessions)}")
        
        # Check attendance
        attendances = db.query(Attendance).all()
        print(f"\nTotal attendance records: {len(attendances)}")
        present = len([a for a in attendances if a.status == 'present'])
        absent = len([a for a in attendances if a.status == 'absent'])
        late = len([a for a in attendances if a.status == 'late'])
        print(f"Present: {present}, Absent: {absent}, Late: {late}")
        
        # Check face embeddings
        from models.database import FaceEmbedding
        embeddings = db.query(FaceEmbedding).all()
        print(f"\nTotal face embeddings: {len(embeddings)}")
        
    finally:
        db.close()

if __name__ == "__main__":
    verify_data()