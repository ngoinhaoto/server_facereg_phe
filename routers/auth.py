from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.db import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
def register_user():
    # Implementation
    pass

@router.post("/login")
def login_user():
    # Implementation
    pass