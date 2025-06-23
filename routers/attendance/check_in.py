from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, BackgroundTasks
from sqlalchemy.orm import Session
from database.db import get_db
from services.face_recognition import FaceRecognitionService
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import ClassSession, Attendance, User, AttendanceStatus
from datetime import datetime, timezone
from starlette.concurrency import run_in_threadpool
from utils.logging import logger
from config.face_recognition_config import face_recognition_config

router = APIRouter()

@router.post("/check-in")
async def check_in(
    session_id: int,
    file: UploadFile = File(...),
    model: str = None,
    try_both_models: bool = True,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    # Use configuration if parameters aren't provided
    if model is None:
        model = face_recognition_config.get_model_for_operation("check_in")
    
    if model not in ["insightface", "deepface"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model selection. Choose 'insightface' or 'deepface'"
        )
    
    if model == "deepface":
        try:
            import deepface
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="DeepFace model is not installed on the server. Please contact administrator."
            )
    
    session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class session not found"
        )
    
    image_data = await file.read()
    
    # Extract face embedding with the primary model
    face_service = FaceRecognitionService.get_instance(model_type=model)
    processed_image = await run_in_threadpool(
        lambda: face_service.preprocess_image(image_data)
    )
    
    # Combine anti-spoofing and face extraction in one call
    # This also checks for face completeness
    result = await run_in_threadpool(
        lambda: face_service.extract_face_embedding(processed_image, check_spoofing=face_recognition_config.ENABLE_ANTISPOOFING)
    )
    
    # Check if we got 3 or 4 values (for backward compatibility)
    if len(result) == 3:
        embedding, confidence, aligned_face = result
        spoof_result = None
    else:
        embedding, confidence, aligned_face, spoof_result = result
    
    if spoof_result:
        if spoof_result.get("is_spoof", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Spoofing detected. Please use a real face for authentication. Score: {spoof_result.get('spoof_score', 0):.2f}"
            )
            
        if spoof_result.get("incomplete_face", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Incomplete face detected: {spoof_result.get('error', 'Please ensure your entire face is visible in the frame.')}"
            )
    
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please try with a clearer photo showing your full face."
        )
    
    # First try matching with the current model
    match, matched_user_id, similarity = await run_in_threadpool(
        lambda: face_service.compare_face(embedding, db, user_id=None, 
                                         threshold=face_recognition_config.SIMILARITY_THRESHOLD)
    )
    
    # If no match found and try_both_models is enabled, try with the other model
    if not match and try_both_models:
        other_model = "insightface" if model == "deepface" else "deepface"
        logger.info(f"No match found with {model}, trying {other_model}")
        
        other_service = FaceRecognitionService.get_instance(model_type=other_model)
        result = await run_in_threadpool(
            lambda: other_service.extract_face_embedding(processed_image)
        )
        
        # Handle different return formats
        if len(result) == 3:
            other_embedding, other_confidence, _ = result
        else:
            other_embedding, other_confidence, _, _ = result
        
        if other_embedding is not None:
            match, matched_user_id, similarity = await run_in_threadpool(
                lambda: other_service.compare_face(other_embedding, db, user_id=None,
                                                 threshold=face_recognition_config.SIMILARITY_THRESHOLD)
            )
            
            if match:
                logger.info(f"Match found with alternate model {other_model}, similarity: {similarity:.2f}")
                # Update the active model and embedding for the rest of the function
                model = other_model
                face_service = other_service
                embedding = other_embedding
                confidence = other_confidence
    
    # If no match found and current user has no embeddings, allow first-time registration
    if not match and current_user.role == "student":
        embeddings_count = await run_in_threadpool(
            lambda: face_service.get_user_embeddings_count(db, current_user.id)
        )
        
        if embeddings_count == 0:
            logger.info(f"First-time face registration for user {current_user.id}")
            # Store the embedding
            await run_in_threadpool(
                lambda: face_service.store_face_embedding(
                    db, current_user.id, embedding, confidence, "attendance"
                )
            )
            # Use the current user since this is their first registration
            matched_user_id = current_user.id
            match = True
    
    # If still no match, authentication failed
    if not match:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Face verification failed. No matching user found (best similarity: {similarity:.2f})."
        )
    
    # Now get the student user who matched
    student_user = db.query(User).filter(User.id == matched_user_id).first()
    if not student_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matched user not found in database."
        )
    
    # Check if the matched student has access to this class
    student_has_access = any(c.id == session.class_id for c in student_user.classes)
    
    # Admin can check in any student, teacher can check in their students
    allowed_to_check_in = False
    if current_user.id == matched_user_id:
        # Users can check themselves in only if enrolled
        allowed_to_check_in = student_has_access
    elif current_user.role == "admin":
        # Admins can check in any student to any class
        allowed_to_check_in = True
        # Optionally, add the student to the class if they're not enrolled
        if not student_has_access:
            logger.info(f"Admin {current_user.id} checking in non-enrolled student {matched_user_id} to class {session.class_id}")
    elif current_user.role == "teacher":
        # Teachers can check in students for classes they teach (student must be enrolled)
        teacher = db.query(User).filter(User.id == current_user.id).first()
        teaches_class = any(c.id == session.class_id for c in teacher.teaching_classes)
        allowed_to_check_in = teaches_class and student_has_access
    
    if not allowed_to_check_in:
        # Be careful with error messages to avoid revealing too much information
        if current_user.id == matched_user_id:
            detail = "You are not enrolled in this class."
        else:
            detail = "You are not authorized to check in this student."
            
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )
    
    # Check if attendance record already exists
    existing_attendance = db.query(Attendance).filter(
        Attendance.student_id == matched_user_id,
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

    if existing_attendance:
        # Update existing attendance
        existing_attendance.status = attendance_status
        existing_attendance.check_in_time = now
        existing_attendance.late_minutes = late_minutes
    else:
        # Create new attendance record
        attendance = Attendance(
            student_id=matched_user_id,
            session_id=session_id,
            status=attendance_status,
            check_in_time=now,
            late_minutes=late_minutes
        )
        db.add(attendance)
    
    # Optionally store this new face to improve recognition if it's the student's own face
    if match and similarity < 0.85 and similarity > 0.65 and current_user.id == matched_user_id:
        embeddings_count = await run_in_threadpool(
            lambda: face_service.get_user_embeddings_count(db, matched_user_id)
        )
        
        if embeddings_count < 10 and background_tasks:
            background_tasks.add_task(
                face_service.store_face_embedding,
                db=db, 
                user_id=matched_user_id,
                embedding=embedding,
                confidence=confidence,
                device_id="auto_update"
            )
    
    db.commit()
    
    class_info = session.class_obj

    return {
        "message": "Attendance recorded successfully",
        "status": attendance_status,
        "late_minutes": late_minutes if attendance_status == AttendanceStatus.LATE.value else 0,
        "face_match_confidence": similarity,
        "admin_user": {
            "id": current_user.id,
            "name": current_user.full_name,
            "username": current_user.username,
            "role": current_user.role
        } if current_user.id != matched_user_id else None,
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
        "check_in_time": now
    }