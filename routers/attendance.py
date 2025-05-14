from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from sqlalchemy.orm import Session
from database.db import get_db
from services.face_recognition import FaceRecognitionService
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import ClassSession, Attendance, User, AttendanceStatus
from datetime import datetime
from typing import Optional


# PLACEHOLDER

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/check-in")
async def check_in(
    session_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Process the image, identify student, and mark attendance.
    This is a placeholder for the ConcreteML face recognition implementation.
    """
    # Just a placeholder - in reality, would process the encrypted face data
    # For now, we'll just mark the current user as present
    
    session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class session not found"
        )
    
    # Check if the user is a student
    if current_user.role != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can check in to classes"
        )
    
    # Check if the student is registered for the class
    student = db.query(User).filter(User.id == current_user.id).first()
    if not any(c.id == session.class_id for c in student.classes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not registered for this class"
        )
    
    # Check if attendance record already exists
    existing_attendance = db.query(Attendance).filter(
        Attendance.student_id == current_user.id,
        Attendance.session_id == session_id
    ).first()
    
    now = datetime.now()
    # Calculate late minutes if student is late
    late_minutes = 0
    status = AttendanceStatus.PRESENT.value
    if now > session.start_time:
        # Calculate minutes late
        time_diff = now - session.start_time
        late_minutes = int(time_diff.total_seconds() / 60)
        if late_minutes > 0:
            status = AttendanceStatus.LATE.value
    
    if existing_attendance:
        # Update existing attendance
        existing_attendance.status = status
        existing_attendance.check_in_time = now
        existing_attendance.late_minutes = late_minutes
    else:
        # Create new attendance record
        attendance = Attendance(
            student_id=current_user.id,
            session_id=session_id,
            status=status,
            check_in_time=now,
            late_minutes=late_minutes
        )
        db.add(attendance)
    
    db.commit()
    
    return {
        "message": "Attendance recorded successfully",
        "status": status,
        "late_minutes": late_minutes if status == AttendanceStatus.LATE.value else 0
    }

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
            "status": attendance.status,
            "check_in_time": attendance.check_in_time,
            "late_minutes": attendance.late_minutes
        })
    
    return result