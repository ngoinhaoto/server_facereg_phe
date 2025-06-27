from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from database.db import get_db
from schemas.class_schema import ClassCreate, ClassResponse, ClassUpdate, ClassWithTeacherResponse
from schemas.user import UserResponse
from crud.class_crud import create_class, get_class, get_classes, update_class, delete_class
from security.auth import get_current_active_user, get_current_teacher_or_admin
from starlette.concurrency import run_in_threadpool
from models.database import Class, User

router = APIRouter(tags=["Class Management"])

@router.post("/", response_model=ClassResponse, status_code=status.HTTP_201_CREATED)
async def create_class_endpoint(
    class_obj: ClassCreate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """Create a new class. Only teachers and admins can create classes."""
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

@router.get("/", response_model=List[ClassWithTeacherResponse])
async def read_classes(
    skip: int = 0, 
    limit: int = 100,
    teacher_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get all classes. Optionally filter by teacher_id."""
    if current_user.role == "teacher":
        teacher_id = current_user.id
    
    # Get classes from the database
    classes = await run_in_threadpool(
        lambda: get_classes(db, skip=skip, limit=limit, teacher_id=teacher_id)
    )
    
    # Enhance classes with teacher information
    result = []
    for class_obj in classes:
        # Convert to dict for easier manipulation
        class_dict = {
            "id": class_obj.id,
            "class_code": class_obj.class_code,
            "name": class_obj.name,
            "description": class_obj.description,
            "semester": class_obj.semester,
            "academic_year": class_obj.academic_year,
            "teacher_id": class_obj.teacher_id,
            "location": class_obj.location,
            "start_time": class_obj.start_time,
            "end_time": class_obj.end_time,
            "created_at": class_obj.created_at,
            "updated_at": class_obj.updated_at,
            "teacher": None
        }
        
        # Add teacher information if available
        if class_obj.teacher_id:
            teacher = db.query(User).filter(User.id == class_obj.teacher_id).first()
            if teacher:
                class_dict["teacher"] = {
                    "id": teacher.id,
                    "full_name": teacher.full_name,
                    "username": teacher.username
                }
        
        result.append(class_dict)
    
    return result

@router.get("/{class_id}", response_model=ClassWithTeacherResponse)
async def read_class(
    class_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get a specific class by ID."""
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
    
    response = {
        "id": db_class.id,
        "class_code": db_class.class_code,
        "name": db_class.name,
        "description": db_class.description,
        "semester": db_class.semester,
        "academic_year": db_class.academic_year,
        "teacher_id": db_class.teacher_id,
        "location": db_class.location,
        "start_time": db_class.start_time,
        "end_time": db_class.end_time,
        "created_at": db_class.created_at,
        "updated_at": db_class.updated_at,
        "teacher": None
    }
    
    if db_class.teacher_id:
        teacher = db.query(User).filter(User.id == db_class.teacher_id).first()
        if teacher:
            response["teacher"] = {
                "id": teacher.id,
                "full_name": teacher.full_name,
                "username": teacher.username
            }
    
    return response

@router.put("/{class_id}", response_model=ClassResponse)
async def update_class_endpoint(
    class_id: int, 
    class_obj: ClassUpdate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """Update class information. Only teachers of the class and admins can update."""
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
    """Delete a class. Only teachers of the class and admins can delete."""
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

@router.get("/me/classes", response_model=List[ClassResponse])
async def get_my_classes(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get classes for the current user (student or teacher)."""
    if current_user.role == "student":
        return current_user.classes
    elif current_user.role == "teacher":
        # Return classes the teacher teaches
        return db.query(Class).filter(Class.teacher_id == current_user.id).all()
    else:
        return []