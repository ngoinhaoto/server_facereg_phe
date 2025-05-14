from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime


class ClassBase(BaseModel):
    class_code: str
    name: str
    description: Optional[str] = None
    semester: str
    academic_year: str
    teacher_id: int
    location: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    students: List[int] = []  # List of student IDs

class ClassCreate(ClassBase):
    pass

class ClassUpdate(ClassBase):
    class_code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    semester: Optional[str] = None
    academic_year: Optional[str] = None
    teacher_id: Optional[int] = None
    location: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    students: List[int] = []  # List of student IDs

class ClassResponse(ClassBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ClassSessionBase(BaseModel):
    class_code: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    notes: Optional[str] = None
    students: List[int] = []  # List of student IDs

class ClassSessionCreate(ClassSessionBase):
    pass

class ClassSessionUpdate(ClassSessionBase):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    students: List[int] = []  

class ClassSessionResponse(ClassSessionBase):
    id: int

    model_config = {
        "from_attributes": True
    }
        