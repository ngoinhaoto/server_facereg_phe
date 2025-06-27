from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import Dict, List
from database.db import get_db
from schemas.class_schema import (
    ClassSessionCreate, ClassSessionResponse, ClassSessionUpdate
)
from schemas.user import UserResponse
from crud.class_crud import (
    get_class, get_session, get_class_sessions, 
    create_class_session, update_class_session, delete_class_session
)
from security.auth import get_current_active_user, get_current_teacher_or_admin
from models.database import User

router = APIRouter(tags=["Class Sessions"])

@router.post("/sessions", response_model=ClassSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session: ClassSessionCreate, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_teacher_or_admin)
):
    """Create a new class session. Only teachers of the class and admins can create sessions."""
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

@router.get("/sessions/{session_id}", response_model=dict)
async def read_session(
    session_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get a specific class session by ID with teacher details."""
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
    
    # Convert session to dict (instead of using undefined schema)
    session_data = {
        "id": db_session.id,
        "class_id": db_session.class_id,
        "session_date": db_session.session_date,
        "start_time": db_session.start_time,
        "end_time": db_session.end_time,
        "notes": db_session.notes,
        # Add class information
        "class_name": db_class.name,
        "class_code": db_class.class_code,
        "location": db_class.location,
        # Add teacher placeholder
        "teacher_id": db_class.teacher_id,
        "teacher_name": None,
        "teacher_username": None,
        "teacher": None 
    }
    
    # Get teacher information if available
    if db_class.teacher_id:
        teacher = db.query(User).filter(User.id == db_class.teacher_id).first()
        if teacher:
            session_data["teacher"] = {
                "id": teacher.id,
                "name": teacher.full_name,
                "username": teacher.username,
                "role": teacher.role
            }
        else:
            # If teacher ID exists but teacher not found, provide partial info
            session_data["teacher"] = {
                "id": db_class.teacher_id,
                "name": "Unknown Teacher",
                "username": None
            }
    
    return session_data

@router.get("/{class_id}/sessions", response_model=List[ClassSessionResponse])
async def read_class_sessions(
    class_id: int, 
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get all sessions for a specific class."""
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
    """Update a class session. Only teachers of the class and admins can update sessions."""
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
    """Delete a class session. Only teachers of the class and admins can delete sessions."""
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

@router.get("/batch-sessions", response_model=Dict[str, List[ClassSessionResponse]])
async def get_multiple_class_sessions(
    class_ids: str = Query(..., description="Comma-separated list of class IDs"),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get sessions for multiple classes in a single request."""
    try:
        id_list = [int(id.strip()) for id in class_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid class ID format")
    
    result = {}
    
    for class_id in id_list:
        db_class = get_class(db, class_id=class_id)
        if not db_class:
            continue
        
        # Check permissions
        if current_user.role == "teacher" and db_class.teacher_id != current_user.id:
            continue
            
        result[str(class_id)] = get_class_sessions(db, class_id=class_id)
    
    return result