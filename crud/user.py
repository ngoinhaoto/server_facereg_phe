from sqlalchemy.orm import Session
from models.database import User
from schemas.user import UserCreate, UserUpdate, UserResponse
from typing import List, Optional
from security.password import get_password_hash, verify_password
import datetime
def get_user(db: Session, user_id: int) -> UserResponse:
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> UserResponse:
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> UserResponse:
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100, role: Optional[str] = None):
    query = db.query(User)
    if role:
        query = query.filter(User.role == role)
    return query.offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> UserResponse:
    now = datetime.datetime.now()
    
    # Set the appropriate ID field based on role
    student_id = None
    staff_id = None
    
    if user.role == "student":
        student_id = user.student_id
    elif user.role in ["teacher", "admin"]:
        staff_id = user.staff_id
    
    db_user = User(
        username=user.username,
        email=user.email,
        role=user.role,
        hashed_password=get_password_hash(user.password),
        is_active=True,
        created_at=now,
        updated_at=now,
        student_id=student_id,
        staff_id=staff_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: UserUpdate) -> UserResponse:
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        return None
    
    update_data = user.model_dump(exclude_unset=True)
    
    # Handle password separately
    if "password" in update_data and update_data["password"]:
        update_data["hashed_password"] = get_password_hash(update_data["password"])
        del update_data["password"]
    
    # Update the model fields
    for key, value in update_data.items():
        if hasattr(db_user, key):  # Only update fields that exist in the model
            setattr(db_user, key, value)
    
    # Commit the changes
    try:
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        print(f"Error updating user: {e}")
        raise

def delete_user(db: Session, user_id: int):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        return False
    
    db.delete(db_user)
    db.commit()
    return True

def authenticate_user(db: Session, username: str, password: str) -> Optional[UserResponse]:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

# Add these new functions to look up users by student/staff ID
def get_user_by_student_id(db: Session, student_id: str) -> UserResponse:
    return db.query(User).filter(User.student_id == student_id).first()

def get_user_by_staff_id(db: Session, staff_id: str) -> UserResponse:
    return db.query(User).filter(User.staff_id == staff_id).first()