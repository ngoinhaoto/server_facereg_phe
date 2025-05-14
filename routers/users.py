from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from database.db import get_db
from schemas.user import UserCreate, UserResponse, UserUpdate
from crud.user import (
    create_user, get_user, get_users, update_user, 
    delete_user, get_user_by_email, get_user_by_username
)
from security.auth import get_current_active_user, get_current_admin_user

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user with email already exists
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username is taken
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    return create_user(db=db, user=user)

@router.get("/", response_model=List[UserResponse])
def read_users(
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
def update_user_info(
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
def delete_user_endpoint(
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