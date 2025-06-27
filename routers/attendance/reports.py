from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.db import get_db
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import ClassSession, Attendance, User
from typing import List

router = APIRouter()

@router.get("/sessions/{session_id}/students")
async def get_session_attendance(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get attendance records for a specific class session.
    """
    session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class session not found"
        )
    
    # Only teachers of this class or admins can view attendance
    class_obj = session.class_obj
    if current_user.role == "teacher" and class_obj.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    elif current_user.role == "student" and not any(c.id == session.class_id for c in current_user.classes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    attendances = session.attendances
    result = []
    
    for attendance in attendances:
        result.append({
            "student_id": attendance.student_id,
            "username": attendance.student.username,
            "full_name": attendance.student.full_name,
            "status": attendance.status,
            "check_in_time": attendance.check_in_time,
            "late_minutes": attendance.late_minutes
        })
    
    return result

@router.get("/student/{student_id}")
async def get_student_attendance(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get attendance records for a specific student."""
    if current_user.role != "admin" and current_user.id != student_id:
        if current_user.role != "teacher":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
            # For teachers, verify they teach at least one class the student is in
            student = db.query(User).filter(User.id == student_id).first()
            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student not found"
                )
                
            # Check if the teacher teaches any class the student is in
            teacher_classes = [c.id for c in current_user.teaching_classes]
            student_classes = [c.id for c in student.classes]
            if not any(c_id in teacher_classes for c_id in student_classes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions"
                )
    
    attendance_records = db.query(Attendance).filter(Attendance.student_id == student_id).all()
    
    result = []
    for record in attendance_records:
        session = record.session
        if session:
            class_obj = session.class_obj
            if class_obj:
                result.append({
                    "id": record.id,
                    "session_id": session.id,
                    "session_date": session.session_date,
                    "start_time": session.start_time,
                    "end_time": session.end_time,
                    "class_id": class_obj.id,
                    "class_name": class_obj.name,
                    "class_code": class_obj.class_code,
                    "status": record.status,
                    "check_in_time": record.check_in_time,
                    "late_minutes": record.late_minutes,
                    "created_at": record.created_at
                })
    
    # Sort by session date, most recent first
    result.sort(key=lambda x: x["session_date"], reverse=True)
    
    return result

@router.post("/sessions/batch/students")
async def get_multiple_sessions_attendance(
    session_ids: List[int],
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get attendance records for multiple class sessions in one request."""
    if current_user.role != "admin" and current_user.role != "teacher":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    result = {}
    
    for session_id in session_ids:
        session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
        if not session:
            result[session_id] = {"error": "Session not found"}
            continue
            
        # Check permissions
        class_obj = session.class_obj
        if current_user.role == "teacher" and class_obj.teacher_id != current_user.id:
            result[session_id] = {"error": "Not enough permissions"}
            continue
        
        # Get attendance records
        attendances = session.attendances
        session_result = []
        
        for attendance in attendances:
            session_result.append({
                "student_id": attendance.student_id,
                "username": attendance.student.username,
                "full_name": attendance.student.full_name,
                "status": attendance.status,
                "check_in_time": attendance.check_in_time,
                "late_minutes": attendance.late_minutes
            })
        
        result[session_id] = session_result
    
    return result