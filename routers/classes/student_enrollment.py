from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database.db import get_db
from schemas.user import UserResponse
from crud.class_crud import (
    get_class, register_student_to_class, 
    remove_student_from_class, get_class_students
)
from security.auth import get_current_active_user, get_current_teacher_or_admin
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["Class Student Enrollment"])

@router.post("/{class_id}/students/{student_id}", status_code=status.HTTP_200_OK)
async def register_student(
    class_id: int,
    student_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """Register a student to a class. Only teachers of the class and admins can register students."""
    db_class = await run_in_threadpool(
        lambda: get_class(db, class_id=class_id)
    )
    
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # If the user is a teacher, they can only register students to their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    success = await run_in_threadpool(
        lambda: register_student_to_class(db, class_id=class_id, student_id=student_id)
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Student not found or not a valid student"
        )
    
    return {"message": "Student registered successfully"}

@router.delete("/{class_id}/students/{student_id}", status_code=status.HTTP_200_OK)
async def remove_student(
    class_id: int,
    student_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """Remove a student from a class. Only teachers of the class and admins can remove students."""
    db_class = await run_in_threadpool(
        lambda: get_class(db, class_id=class_id)
    )
    
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # If the user is a teacher, they can only remove students from their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    success = await run_in_threadpool(
        lambda: remove_student_from_class(db, class_id=class_id, student_id=student_id)
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Student not found or not registered in this class"
        )
    
    return {"message": "Student removed successfully"}

@router.get("/{class_id}/students", response_model=List[UserResponse])
async def get_class_students_endpoint(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get all students registered for a specific class."""
    db_class = await run_in_threadpool(
        lambda: get_class(db, class_id=class_id)
    )
    
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # Teachers can only see students in their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    students = await run_in_threadpool(
        lambda: get_class_students(db, class_id=class_id)
    )
    
    return students