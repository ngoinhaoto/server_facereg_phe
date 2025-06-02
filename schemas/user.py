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
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)  # Add this line
    role: UserRole
    student_id: Optional[str] = None
    staff_id: Optional[str] = None
    
        
class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    password_confirmation: str = Field(..., min_length=8, max_length=128)
    
    # Pydantic v2 validation format
    @field_validator('password_confirmation')
    def passwords_match(cls, v, values):
        if 'password' in values.data and v != values.data['password']:
            raise ValueError('Passwords do not match')
        return v
    
    # Add a validator to ensure the right ID field is used based on role
    @field_validator('student_id', 'staff_id')
    def validate_ids(cls, v, info):
        field_name = info.field_name
        if 'role' in info.data:
            role = info.data['role']
            if field_name == 'student_id' and role != 'student' and v is not None:
                raise ValueError('Only students should have a student ID')
            if field_name == 'staff_id' and role == 'student' and v is not None:
                raise ValueError('Students should not have a staff ID')
        return v

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)  # Make sure this line exists
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    student_id: Optional[str] = None
    staff_id: Optional[str] = None
class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None 

    model_config = {
        "from_attributes": True
    }

