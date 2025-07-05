from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel
class ClassBase(BaseModel):
    class_code: str
    name: str
    description: Optional[str] = None
    semester: str
    academic_year: str
    teacher_id: Optional[int] = None  # Make this optional with None as default
    location: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # students: List[int] = []  # List of student IDs

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
    # students: List[int] = []  # List of student IDs

class ClassSessionBase(BaseModel):
    class_id: int  # This should be the correct field name, not class_code
    session_date: datetime
    start_time: datetime
    end_time: datetime
    notes: Optional[str] = None

class ClassSessionCreate(ClassSessionBase):
    pass

class ClassSessionUpdate(ClassSessionBase):
    class_id: Optional[int] = None  # Make optional for updates
    session_date: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    notes: Optional[str] = None

class ClassSessionResponse(ClassSessionBase):
    id: int

    model_config = {
        "from_attributes": True
    }


# Add this at the end of your file
class TeacherInfo(BaseModel):
    id: int
    full_name: Optional[str] = None
    username: Optional[str] = None
    
    model_config = {
        "from_attributes": True
    }

class ClassResponse(ClassBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    sessions: List[ClassSessionResponse] = []  

    model_config = {
        "from_attributes": True
    }

class ClassWithTeacherResponse(ClassResponse):
    teacher: Optional[TeacherInfo] = None
