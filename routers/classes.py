from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from database.db import get_db
from schemas.class_schema import (
    ClassCreate, ClassResponse, ClassUpdate, 
    ClassSessionCreate, ClassSessionResponse, ClassSessionUpdate
)
from schemas.user import UserResponse
from crud.class_crud import (
    create_class, get_class, get_classes, update_class, delete_class,
    register_student_to_class, remove_student_from_class, get_class_students,
    get_class_sessions, get_session, create_class_session, update_class_session, delete_class_session
)
from security.auth import get_current_active_user, get_current_teacher_or_admin
from starlette.concurrency import run_in_threadpool  # Add this import
from models.database import Class, User

# Add this new model to your imports
class ClassWithStudentsResponse(ClassResponse):
    students: List[UserResponse] = []

router = APIRouter(prefix="/classes", tags=["Classes"])

@router.post("/", response_model=ClassResponse, status_code=status.HTTP_201_CREATED)
async def create_class_endpoint(
    class_obj: ClassCreate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Create a new class. Only teachers and admins can create classes.
    """
    # If the user is a teacher, they can only create classes where they are the teacher
    if current_user.role == "teacher" and class_obj.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teachers can only create classes where they are the teacher"
        )
    
    # Check if class with the same code already exists
    existing_class = await run_in_threadpool(
        lambda: db.query(Class).filter(Class.class_code == class_obj.class_code).first()
    )
    
    if existing_class:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Class with code '{class_obj.class_code}' already exists"
        )
    
    # Run the database operation in a thread pool
    return await run_in_threadpool(lambda: create_class(db=db, class_obj=class_obj))

@router.get("/", response_model=List[ClassResponse])
async def read_classes(
    skip: int = 0, 
    limit: int = 100,
    teacher_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get all classes. Optionally filter by teacher_id.
    """
    if current_user.role == "teacher":
        teacher_id = current_user.id
    
    classes = await run_in_threadpool(
        lambda: get_classes(db, skip=skip, limit=limit, teacher_id=teacher_id)
    )
    return classes

@router.get("/{class_id}", response_model=ClassResponse)
async def read_class(
    class_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get a specific class by ID.
    """
    db_class = get_class(db, class_id=class_id)
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return db_class

@router.put("/{class_id}", response_model=ClassResponse)
async def update_class_endpoint(
    class_id: int, 
    class_obj: ClassUpdate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Update class information. Only teachers of the class and admins can update.
    """
    db_class = get_class(db, class_id=class_id)
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # If the user is a teacher, they can only update their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # If changing teacher_id, verify the user has permission
    if class_obj.teacher_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can change the teacher of a class"
        )
    
    updated_class = update_class(db, class_id=class_id, class_obj=class_obj)
    return updated_class

@router.delete("/{class_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_class_endpoint(
    class_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Delete a class. Only teachers of the class and admins can delete.
    """
    db_class = get_class(db, class_id=class_id)
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # If the user is a teacher, they can only delete their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    success = delete_class(db, class_id=class_id)
    return None

@router.post("/{class_id}/students/{student_id}", status_code=status.HTTP_200_OK)
async def register_student(
    class_id: int,
    student_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Register a student to a class. Only teachers of the class and admins can register students.
    """
    db_class = get_class(db, class_id=class_id)
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
    
    success = register_student_to_class(db, class_id=class_id, student_id=student_id)
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
    """
    Remove a student from a class. Only teachers of the class and admins can remove students.
    """
    db_class = get_class(db, class_id=class_id)
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
    
    success = remove_student_from_class(db, class_id=class_id, student_id=student_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Student not found or not registered in this class"
        )
    
    return {"message": "Student removed successfully"}

# Class sessions endpoints
@router.post("/sessions", response_model=ClassSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session: ClassSessionCreate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Create a new class session. Only teachers of the class and admins can create sessions.
    """
    db_class = get_class(db, class_id=session.class_id)
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # If the user is a teacher, they can only create sessions for their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return create_class_session(db=db, session=session)

@router.get("/sessions/{session_id}", response_model=ClassSessionResponse)
async def read_session(
    session_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get a specific class session by ID.
    """
    db_session = get_session(db, session_id=session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    
    # If the user is a teacher, they can only see sessions for their own classes
    db_class = get_class(db, class_id=db_session.class_id)
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return db_session

@router.get("/{class_id}/sessions", response_model=List[ClassSessionResponse])
async def read_class_sessions(
    class_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get all sessions for a specific class.
    """
    db_class = get_class(db, class_id=class_id)
    if db_class is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Class not found"
        )
    
    # If the user is a teacher, they can only see sessions for their own classes
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return get_class_sessions(db, class_id=class_id)

@router.put("/sessions/{session_id}", response_model=ClassSessionResponse)
async def update_session(
    session_id: int, 
    session: ClassSessionUpdate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Update a class session. Only teachers of the class and admins can update sessions.
    """
    db_session = get_session(db, session_id=session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    
    # If the user is a teacher, they can only update sessions for their own classes
    db_class = get_class(db, class_id=db_session.class_id)
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    updated_session = update_class_session(db, session_id=session_id, session=session)
    return updated_session

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """
    Delete a class session. Only teachers of the class and admins can delete sessions.
    """
    db_session = get_session(db, session_id=session_id)
    if db_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Session not found"
        )
    
    # If the user is a teacher, they can only delete sessions for their own classes
    db_class = get_class(db, class_id=db_session.class_id)
    if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    success = delete_class_session(db, session_id=session_id)
    return None