from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from sqlalchemy.orm import Session
from database.db import get_db
from services.face_recognition import FaceRecognitionService
from schemas.user import UserResponse
from security.auth import get_current_active_user
from fastapi.security.api_key import APIKeyHeader
from config.app import settings
from models.database import FaceEmbedding, User, ClassSession, Attendance, AttendanceStatus
from starlette.concurrency import run_in_threadpool
import base64  # Add this back
from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.logging import logger
from config.face_recognition_config import face_recognition_config
from services.face_recognition.duplicate_detection import DuplicateFaceDetector
import uuid
from datetime import datetime, timezone
from fastapi import BackgroundTasks
from pydantic import BaseModel

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key == settings.PHE_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key"
    )

router = APIRouter()

@router.post("/register-face")
async def register_face(
    file: UploadFile = File(...),
    device_id: str = "web",
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    # Always use DeepFace model
    model = "deepface"
    
    image_data = await file.read()
    
    logger.info(f"Registering face using {model} model")
    
    face_service = FaceRecognitionService.get_instance(model_type=model)
    processed_image = await run_in_threadpool(
        lambda: face_service.preprocess_image(image_data)
    )
    
    result = await run_in_threadpool(
        lambda: face_service.extract_face_embedding(
            processed_image, 
            check_spoofing=face_recognition_config.ENABLE_REGISTRATION_ANTISPOOFING
        )
    )
    
    if len(result) == 3:
        embedding_primary, confidence_primary, aligned_face_primary = result
        spoof_result = None
    else:
        embedding_primary, confidence_primary, aligned_face_primary, spoof_result = result
    
    # Check for spoofing or incomplete face
    if spoof_result:
        if spoof_result.get("is_spoof", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Spoofing detected. Please use a real face for registration."
            )
        
        if spoof_result.get("incomplete_face", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Incomplete face detected: {spoof_result.get('error', 'Please ensure your entire face is visible in the frame.')}"
            )
    
    if embedding_primary is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please try with a clearer photo showing your full face."
        )
    
    if face_recognition_config.ENABLE_DUPLICATE_DETECTION:
        is_duplicate, duplicate_info = await DuplicateFaceDetector.check_for_duplicates(
            embedding_primary, 
            current_user_id=current_user.id
        )
        
        if is_duplicate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This face appears to be already registered to another user: {duplicate_info.get('duplicate_user_name', 'Unknown')}"
            )
    
    # Continue with the registration process
    registration_group_id = str(uuid.uuid4())
    
    embedding_id_primary = await run_in_threadpool(
        lambda: face_service.store_face_embedding(
            db, current_user.id, embedding_primary, confidence_primary, 
            device_id, model, registration_group_id
        )
    )
    
    # Get total embeddings count
    embeddings_count = await run_in_threadpool(
        lambda: face_service.get_user_embeddings_count(db, current_user.id)
    )
    
    # Return response
    response = {
        "message": "Face registered successfully",
        "embeddings_count": embeddings_count,
        "confidence": confidence_primary,
        "face_id": embedding_id_primary,
        "anti_spoofing_passed": True,
        "duplicate_check_passed": True
    }
    
    # Add aligned face if available
    if aligned_face_primary:
        response["aligned_face"] = base64.b64encode(aligned_face_primary).decode('utf-8')
    
    return response

@router.get("/my-faces", response_model=Dict)
async def get_my_faces(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get information about registered faces for the current user, grouping by registration."""
    embeddings = db.query(FaceEmbedding).filter(
        FaceEmbedding.user_id == current_user.id
    ).order_by(FaceEmbedding.created_at.desc()).all()
    
    grouped_embeddings = {}
    
    for emb in embeddings:
        # Check if this embedding has a registration group
        group_id = getattr(emb, 'registration_group_id', None)
        
        # If no group ID, use the embedding ID as a unique identifier
        if not group_id:
            group_id = f"single_{emb.id}"
        
        # Initialize group if not exists
        if group_id not in grouped_embeddings:
            grouped_embeddings[group_id] = {
                "id": emb.id,  # Use the ID of the first embedding found
                "created_at": emb.created_at,
                "device_id": emb.device_id.split('_auto_')[0] if '_auto_' in emb.device_id else emb.device_id,
                "confidence": emb.confidence_score,
                "models": [],
                "image": None
            }
        
        # Add this model to the group's models list
        model_type = getattr(emb, 'model_type', 'unknown')
        grouped_embeddings[group_id]["models"].append(model_type)
    
    # Convert the grouped dictionary to a list
    result_faces = list(grouped_embeddings.values())
    
    # Sort by created_at (newest first)
    result_faces.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "count": len(result_faces),
        "faces": result_faces
    }

@router.delete("/my-faces/{embedding_id}")
async def delete_face(
    embedding_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Delete all face embeddings from the same registration group."""

    embedding = db.query(FaceEmbedding).filter(
        FaceEmbedding.id == embedding_id,
        FaceEmbedding.user_id == current_user.id
    ).first()
    
    if not embedding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face embedding not found"
        )
    
    has_group_id = hasattr(embedding, 'registration_group_id') and embedding.registration_group_id
    
    if has_group_id:
        embeddings_to_delete = db.query(FaceEmbedding).filter(
            FaceEmbedding.registration_group_id == embedding.registration_group_id,
            FaceEmbedding.user_id == current_user.id
        ).all()
        
        for emb in embeddings_to_delete:
            db.delete(emb)
    else:
        # If no group ID (legacy data), just delete the single embedding
        db.delete(embedding)
    
    db.commit()
    
    return {"message": "Face embedding(s) deleted successfully"}

@router.get("/faces/{embedding_id}", response_model=dict)
async def get_face_details(
    embedding_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get details of a specific face embedding."""
    # First check if the user is an admin or the owner of this embedding
    embedding = db.query(FaceEmbedding).filter(FaceEmbedding.id == embedding_id).first()
    
    if not embedding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face embedding not found"
        )
    
    # Only allow users to view their own face embeddings (or admins)
    if embedding.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this face embedding"
        )
    
    # Get the user this embedding belongs to
    user = db.query(User).filter(User.id == embedding.user_id).first()
    
    # Return the embedding details
    response = {
        "id": embedding.id,
        "user_id": embedding.user_id,
        "username": user.username if user else "Unknown",
        "confidence_score": embedding.confidence_score,
        "device_id": embedding.device_id,
        "created_at": embedding.created_at
    }
    
    return response

@router.get("/face-recognition-settings")
async def get_face_recognition_settings(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get available face recognition settings and capabilities"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can access face recognition settings"
        )
    
    available_models = ["deepface"]  # Remove insightface
    
    try:
        import deepface
        deepface_version = deepface.__version__
    except ImportError:
        deepface_version = None
    
    try:
        import face_recognition
        available_models.append("dlib")
        dlib_available = True
    except ImportError:
        dlib_available = False
    
    return {
        "available_models": available_models,
        "model_versions": {
            "deepface": deepface_version,
            "dlib": "Available" if dlib_available else "Not installed"
        },
        "anti_spoofing_supported": {
            "deepface": "Yes" if deepface_version else "Not installed",
            "dlib": "Limited" if dlib_available else "Not installed"
        }
    }

# Define a model for the request body
class PHECheckInRequest(BaseModel):
    session_id: int
    user_id: int
    verification_method: str = "phe"

@router.post("/phe-check-in")
async def phe_check_in(
    data: PHECheckInRequest,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key) 
):
    """Record attendance based on PHE verification from microservice"""
    # Extract values from the request body
    session_id = data.session_id
    user_id = data.user_id
    verification_method = data.verification_method
    
    # Validate session
    session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class session not found"
        )
    
    # Validate user
    student_user = db.query(User).filter(User.id == user_id).first()
    if not student_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in database."
        )
    
    # Check if the student has access to this class
    student_has_access = any(c.id == session.class_id for c in student_user.classes)
    
    # For PHE check-ins, we just verify that the student is enrolled in the class
    if not student_has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student is not enrolled in this class."
        )
    
    # Check if attendance record already exists
    existing_attendance = db.query(Attendance).filter(
        Attendance.student_id == user_id,
        Attendance.session_id == session_id
    ).first()
    
    now = datetime.now(timezone.utc)
    # Calculate late minutes if student is late
    late_minutes = 0
    attendance_status = AttendanceStatus.PRESENT.value
    if now > session.start_time:
        # Calculate minutes late
        time_diff = now - session.start_time
        late_minutes = int(time_diff.total_seconds() / 60)
        if late_minutes > 0:
            attendance_status = AttendanceStatus.LATE.value

    try:
        if existing_attendance:
            # Update existing attendance
            existing_attendance.status = attendance_status
            existing_attendance.check_in_time = now
            existing_attendance.late_minutes = late_minutes
            existing_attendance.verification_method = verification_method
        else:
            # Create new attendance record - no similarity needed
            attendance = Attendance(
                student_id=user_id,
                session_id=session_id,
                status=attendance_status,
                check_in_time=now,
                late_minutes=late_minutes,
                verification_method=verification_method
            )
            db.add(attendance)
        
        db.commit()
        
        class_info = session.class_obj

        return {
            "message": "Attendance recorded successfully via PHE",
            "status": attendance_status,
            "late_minutes": late_minutes if attendance_status == AttendanceStatus.LATE.value else 0,
            "user": {
                "id": student_user.id,
                "name": student_user.full_name,
                "username": student_user.username,
                "role": student_user.role
            },
            "session": {
                "id": session.id,
                "date": session.session_date,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "class": {
                    "id": class_info.id,
                    "name": class_info.name,
                    "code": class_info.class_code
                }
            },
            "check_in_time": now,
            "verification_method": verification_method
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error recording PHE attendance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording attendance: {str(e)}"
        )