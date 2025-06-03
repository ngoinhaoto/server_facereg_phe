from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from database.db import get_db
from schemas.user import UserCreate, UserResponse, UserUpdate
from schemas.class_schema import ClassResponse
from models.database import User, Class

from crud.user import (
    create_user, get_user, get_users, update_user, 
    delete_user, get_user_by_email, get_user_by_username,
    get_user_by_student_id, get_user_by_staff_id
)
from security.auth import get_current_active_user, get_current_admin_user
from fastapi.concurrency import run_in_threadpool

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check email and username as before
    db_user = await run_in_threadpool(lambda: get_user_by_email(db, email=user.email))
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    db_user = await run_in_threadpool(lambda: get_user_by_username(db, username=user.username))
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Check student_id if applicable
    if user.role == "student" and user.student_id:
        existing_student = await run_in_threadpool(lambda: get_user_by_student_id(db, student_id=user.student_id))
        if existing_student:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student ID already registered"
            )
    
    # Check staff_id if applicable
    if user.role in ["teacher", "admin"] and user.staff_id:
        existing_staff = await run_in_threadpool(lambda: get_user_by_staff_id(db, staff_id=user.staff_id))
        if existing_staff:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Staff ID already registered"
            )
    
    return await run_in_threadpool(lambda: create_user(db=db, user=user))

@router.get("/", response_model=List[UserResponse])
async def read_users(
    skip: int = 0, 
    limit: int = 100, 
    role: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """
    Get all users. Only accessible by admin users.
    Optionally filter by role.
    """
    users = get_users(db, skip=skip, limit=limit, role=role)
    return users

@router.get("/{user_id}", response_model=UserResponse)
def read_user(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get a specific user by ID.
    Users can access their own information, admins can access anyone's.
    """
    # Regular users can only access their own information
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
        
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )
    return db_user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user_info(
    user_id: int, 
    user: UserUpdate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Update user information.
    Users can update their own information, admins can update anyone's.
    """
    # Regular users can only modify their own information
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check student_id if being updated
    if user.student_id:
        existing_student = await run_in_threadpool(lambda: get_user_by_student_id(db, student_id=user.student_id))
        if existing_student and existing_student.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student ID already registered"
            )
    
    # Check staff_id if being updated
    if user.staff_id:
        existing_staff = await run_in_threadpool(lambda: get_user_by_staff_id(db, staff_id=user.staff_id))
        if existing_staff and existing_staff.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Staff ID already registered"
            )
    
    # If updating email, check if it's already taken
    if user.email:
        db_user = get_user_by_email(db, email=user.email)
        if db_user and db_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # If updating username, check if it's already taken
    if user.username:
        db_user = get_user_by_username(db, username=user.username)
        if db_user and db_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Regular users cannot change their role
    if user.role and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can change user roles"
        )
    
    updated_user = update_user(db, user_id=user_id, user=user)
    if updated_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )
    
    return updated_user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_endpoint(
    user_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """
    Delete a user. Only admins can delete users.
    """
    success = delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )
    return None

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get the currently authenticated user's information.
    This endpoint is accessible by all authenticated users regardless of role.
    """
    return current_user

@router.get("/{user_id}/classes", response_model=List[ClassResponse])
async def get_user_classes(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get classes a user is enrolled in or teaches"""
    # Check permissions
    if current_user.id != user_id and current_user.role not in ["admin", "teacher"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Not authorized to view this user's classes"
        )
    
    # Get the user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.role == "student":
        # Return classes the student is enrolled in
        return user.classes
    elif user.role == "teacher":
        # Return classes the teacher teaches
        return db.query(Class).filter(Class.teacher_id == user.id).all()
    else:
        return []