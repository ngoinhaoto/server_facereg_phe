from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from enum import Enum

class UserRole(str, Enum):
    student = "student"
    teacher = "teacher"
    admin = "admin"

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: UserRole

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    password_confirmation: str = Field(..., min_length=8, max_length=128)
    
    # Pydantic v2 validation format
    @field_validator('password_confirmation')
    def passwords_match(cls, v, values):
        if 'password' in values.data and v != values.data['password']:
            raise ValueError('Passwords do not match')
        return v

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None 

    model_config = {
        "from_attributes": True
    }
