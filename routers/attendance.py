from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, BackgroundTasks
from sqlalchemy.orm import Session
from database.db import get_db
from services.face_recognition import FaceRecognitionService
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import ClassSession, Attendance, User, AttendanceStatus, FaceEmbedding
from datetime import datetime
from typing import Optional, List
from starlette.concurrency import run_in_threadpool
import base64
from utils.logging import logger

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/check-in")
async def check_in(
    session_id: int,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Check in to a class session using face recognition."""
    # Check if the session exists
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
    
    # Read the image file
    image_data = await file.read()
    
    # Initialize face recognition service
    face_service = FaceRecognitionService()
    
    # Preprocess the image
    processed_image = await run_in_threadpool(
        lambda: face_service.preprocess_image(image_data)
    )
    
    # Extract face embedding
    embedding, confidence, _ = await run_in_threadpool(
        lambda: face_service.extract_face_embedding(processed_image)
    )
    
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please try with a clearer photo."
        )
    
    # Verify the face matches the user
    match, matched_user_id, similarity = await run_in_threadpool(
        lambda: face_service.compare_face(embedding, db, current_user.id)
    )
    
    # Get the number of stored embeddings for the user
    embeddings_count = await run_in_threadpool(
        lambda: face_service.get_user_embeddings_count(db, current_user.id)
    )
    
    # If user has no face embeddings, store this one automatically
    if embeddings_count == 0:
        logger.info(f"First-time face registration for user {current_user.id}")
        # Store the embedding
        await run_in_threadpool(
            lambda: face_service.store_face_embedding(
                db, current_user.id, embedding, confidence, "attendance"
            )
        )
        # Skip verification for first registration
        match = True
    elif not match:
        # User has embeddings but face doesn't match
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Face verification failed (similarity score: {similarity:.2f}). Please try again."
        )
    
    # Check if attendance record already exists
    existing_attendance = db.query(Attendance).filter(
        Attendance.student_id == current_user.id,
        Attendance.session_id == session_id
    ).first()
    
    now = datetime.now()
    # Calculate late minutes if student is late
    late_minutes = 0
    attendance_status = AttendanceStatus.PRESENT.value
    if now > session.start_time:
        # Calculate minutes late
        time_diff = now - session.start_time
        late_minutes = int(time_diff.total_seconds() / 60)
        if late_minutes > 0:
            attendance_status = AttendanceStatus.LATE.value

    if existing_attendance:
        # Update existing attendance
        existing_attendance.status = attendance_status
        existing_attendance.check_in_time = now
        existing_attendance.late_minutes = late_minutes
    else:
        # Create new attendance record
        attendance = Attendance(
            student_id=current_user.id,
            session_id=session_id,
            status=attendance_status,
            check_in_time=now,
            late_minutes=late_minutes
        )
        db.add(attendance)
    
    if match and similarity < 0.85 and embeddings_count < 10 and similarity > 0.65:
        if background_tasks:
            background_tasks.add_task(
                face_service.store_face_embedding,
                db=db, 
                user_id=current_user.id,
                embedding=embedding,
                confidence=confidence,
                device_id="auto_update"
            )
    
    db.commit()
    
    return {
        "message": "Attendance recorded successfully",
        "status": attendance_status,
        "late_minutes": late_minutes if attendance_status == AttendanceStatus.LATE.value else 0,
        "face_match_confidence": similarity
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

@router.post("/register-face")
async def register_face(
    file: UploadFile = File(...),
    device_id: str = "web",
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    face_service = FaceRecognitionService()
    
    image_data = await file.read()
    
    processed_image = await run_in_threadpool(
        lambda: face_service.preprocess_image(image_data)
    )
    
    embedding, confidence, aligned_face = await run_in_threadpool(
        lambda: face_service.extract_face_embedding(processed_image)
    )
    
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please try with a clearer photo."
        )
    
    embeddings_count = await run_in_threadpool(
        lambda: face_service.get_user_embeddings_count(db, current_user.id)
    )
    
    success = await run_in_threadpool(
        lambda: face_service.store_face_embedding(
            db, current_user.id, embedding, confidence, device_id
        )
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store face embedding"
        )
    
    # Return response with aligned face preview if available
    response = {
        "message": "Face registered successfully",
        "embeddings_count": embeddings_count + 1,
        "confidence": confidence
    }
    
    if aligned_face:
        response["aligned_face"] = base64.b64encode(aligned_face).decode('utf-8')
    
    return response

@router.get("/my-faces")
async def get_my_faces(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get information about registered faces for the current user."""
    embeddings = db.query(FaceEmbedding).filter(
        FaceEmbedding.user_id == current_user.id
    ).all()
    
    return {
        "count": len(embeddings),
        "faces": [
            {
                "id": emb.id,
                "created_at": emb.created_at,
                "device_id": emb.device_id,
                "confidence": emb.confidence_score
            }
            for emb in embeddings
        ]
    }

@router.delete("/my-faces/{embedding_id}")
async def delete_face(
    embedding_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Delete a face embedding."""
    embedding = db.query(FaceEmbedding).filter(
        FaceEmbedding.id == embedding_id,
        FaceEmbedding.user_id == current_user.id
    ).first()
    
    if not embedding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face embedding not found"
        )
    
    db.delete(embedding)
    db.commit()
    
    return {"message": "Face embedding deleted successfully"}