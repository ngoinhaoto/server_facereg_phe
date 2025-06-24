from sqlalchemy.orm import Session
from models.database import Class, ClassSession, User, Attendance, AttendanceStatus
from schemas.class_schema import ClassCreate, ClassUpdate, ClassSessionCreate, ClassSessionUpdate
from typing import Optional, List

def get_class(db: Session, class_id: int) -> Optional[Class]:
    return db.query(Class).filter(Class.id == class_id).first()

def get_class_by_code(db: Session, class_code: str) -> Optional[Class]:
    return db.query(Class).filter(Class.class_code == class_code).first()


def get_classes(db: Session, skip: int = 0, limit: int = 100, teacher_id: Optional[int] = None):
    query = db.query(Class)
    if teacher_id:
        query = query.filter(Class.teacher_id == teacher_id)
    return query.offset(skip).limit(limit).all()

def create_class(db: Session, class_obj: ClassCreate) -> Class:
    db_class = Class(
        class_code=class_obj.class_code,
        name=class_obj.name,
        description=class_obj.description,
        semester=class_obj.semester,
        academic_year=class_obj.academic_year,
        teacher_id=class_obj.teacher_id,
        location=class_obj.location,
        start_time=class_obj.start_time,
        end_time=class_obj.end_time
    )
    db.add(db_class)
    db.commit()
    db.refresh(db_class)
    return db_class

def update_class(db: Session, class_id: int, class_obj: ClassUpdate):
    db_class = db.query(Class).filter(Class.id == class_id).first()
    if not db_class:
        return None
    
    update_data = class_obj.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_class, key, value)

    db.commit()
    db.refresh(db_class)
    return db_class

def delete_class(db: Session, class_id: int):
    db_class = db.query(Class).filter(Class.id == class_id).first()
    if not db_class:
        return False
    
    db.delete(db_class)
    db.commit()
    return True
    
def register_student_to_class(db: Session, class_id: int, student_id: int):
    db_class = db.query(Class).filter(Class.id == class_id).first()
    db_student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    
    if not db_class or not db_student:
        return False
    
    if db_student in db_class.students:
        return True
    
    # Add student to class
    db_class.students.append(db_student)
    db.commit()
    
    # Create attendance records for all existing sessions
    existing_sessions = db.query(ClassSession).filter(ClassSession.class_id == class_id).all()
    if existing_sessions:
        attendance_records = [
            Attendance(
                student_id=student_id,
                session_id=session.id,
                status=AttendanceStatus.ABSENT.value,
                check_in_time=None,
                late_minutes=0
            )
            for session in existing_sessions
        ]
        
        if attendance_records:
            db.bulk_save_objects(attendance_records)
            db.commit()
    
    return True

def remove_student_from_class(db: Session, class_id: int, student_id: int):
    db_class = db.query(Class).filter(Class.id == class_id).first()
    db_student = db.query(User).filter(User.id == student_id).first()
    
    if not db_class or not db_student:
        return False
    
    if db_student in db_class.students:
        db_class.students.remove(db_student)
        db.commit()
        return True
    
    return False


def get_class_students(db: Session, class_id: int):
    db_class = db.query(Class).filter(Class.id == class_id).first()
    if not db_class:
        return None
    
    return db_class.students

def get_session(db: Session, session_id: int):
    return db.query(ClassSession).filter(ClassSession.id == session_id).first()

def get_class_sessions(db: Session, class_id: int):
    return db.query(ClassSession).filter(ClassSession.class_id == class_id).all()

def create_class_session(db: Session, session: ClassSessionCreate) -> ClassSession:
    try:
        db_session = ClassSession(
            class_id=session.class_id,
            session_date=session.session_date,
            start_time=session.start_time,
            end_time=session.end_time,
            notes=session.notes
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        class_obj = db.query(Class).filter(Class.id == session.class_id).first()
        if class_obj and class_obj.students:
            attendance_records = [
                Attendance(
                    student_id=student.id,
                    session_id=db_session.id,
                    status=AttendanceStatus.ABSENT.value,
                    check_in_time=None,
                    late_minutes=0
                )
                for student in class_obj.students
            ]
            
            if attendance_records:
                db.bulk_save_objects(attendance_records)
                db.commit()
        
        return db_session
    except Exception as e:
        db.rollback()
        # Log the error
        print(f"Error creating class session: {str(e)}")
        raise

def update_class_session(db: Session, session_id: int, session: ClassSessionUpdate):
    db_session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not db_session:
        return None
    
    update_data = session.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(db_session, key, value)
    
    db.commit()
    db.refresh(db_session)
    return db_session

def delete_class_session(db: Session, session_id: int):
    db_session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not db_session:
        return False
    
    db.delete(db_session)
    db.commit()
    return True