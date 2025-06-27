from pydantic import BaseModel
from typing import List
from schemas.user import UserResponse
from schemas.class_schema import ClassResponse

class StudentRegistration(BaseModel):
    student_id: int

class ClassWithStudentsResponse(ClassResponse):
    students: List[UserResponse] = []