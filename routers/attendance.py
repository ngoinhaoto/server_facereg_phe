from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, BackgroundTasks
from sqlalchemy.orm import Session
from database.db import get_db
from services.face_recognition import FaceRecognitionService
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import ClassSession, Attendance, User, AttendanceStatus, FaceEmbedding, FaceImage
from datetime import datetime, timezone
from typing import Optional, List, Dict
from starlette.concurrency import run_in_threadpool
import base64
from utils.logging import logger
from config.face_recognition_config import face_recognition_config

router = APIRouter(prefix="/attendance", tags=["Attendance"])

@router.post("/check-in")
async def check_in(
    session_id: int,
    file: UploadFile = File(...),
    model: str = None,
    try_both_models: bool = True,  # Add this parameter
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
    
    embedding, confidence, aligned_face = await run_in_threadpool(
        lambda: face_service.extract_face_embedding(processed_image)
    )
    
    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please try with a clearer photo."
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
        other_embedding, other_confidence, _ = await run_in_threadpool(
            lambda: other_service.extract_face_embedding(processed_image)
        )
        
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
            "full_name": attendance.student.full_name,  # Add the full name here
            "status": attendance.status,
            "check_in_time": attendance.check_in_time,
            "late_minutes": attendance.late_minutes
        })
    
    return result

@router.post("/register-face")
async def register_face(
    file: UploadFile = File(...),
    device_id: str = "web",
    model: str = None,
    store_both_models: bool = True,  # Add this parameter
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    if model is None:
        model = face_recognition_config.get_model_for_operation("register_face")
    
    if model not in ["insightface", "deepface"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model selection. Choose 'insightface' or 'deepface'"
        )
    
    image_data = await file.read()
    
    logger.info(f"Registering face using {model} model with store_both_models={store_both_models}")
    
    face_service = FaceRecognitionService.get_instance(model_type=model)
    processed_image = await run_in_threadpool(
        lambda: face_service.preprocess_image(image_data)
    )
    
    embedding_primary, confidence_primary, aligned_face_primary = await run_in_threadpool(
        lambda: face_service.extract_face_embedding(processed_image)
    )
    
    if embedding_primary is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image. Please try with a clearer photo."
        )
    
    # Store the primary model embedding
    embedding_id_primary = await run_in_threadpool(
        lambda: face_service.store_face_embedding(
            db, current_user.id, embedding_primary, confidence_primary, device_id, model
        )
    )
    
    # Store the aligned face image if available
    if aligned_face_primary and embedding_id_primary:
        try:
            face_image = FaceImage(
                embedding_id=embedding_id_primary,
                image_data=aligned_face_primary
            )
            db.add(face_image)
            db.commit()
        except Exception as e:
            logger.error(f"Error storing face image: {str(e)}")
    
    # If requested, also store with the secondary model
    embedding_id_secondary = None
    if store_both_models:
        # Get the other model
        other_model = "insightface" if model == "deepface" else "deepface"
        
        try:
            # Process with secondary model
            secondary_service = FaceRecognitionService.get_instance(model_type=other_model)
            embedding_secondary, confidence_secondary, _ = await run_in_threadpool(
                lambda: secondary_service.extract_face_embedding(processed_image)
            )
            
            if embedding_secondary is not None:
                # Store the secondary embedding
                embedding_id_secondary = await run_in_threadpool(
                    lambda: secondary_service.store_face_embedding(
                        db, current_user.id, embedding_secondary, confidence_secondary, 
                        f"{device_id}_auto_{other_model}", other_model
                    )
                )
                logger.info(f"Successfully stored secondary embedding with {other_model}")
        except Exception as e:
            logger.error(f"Error processing with secondary model {other_model}: {str(e)}")
    
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
        "dual_models": store_both_models,
        "secondary_model_success": embedding_id_secondary is not None if store_both_models else None
    }
    
    if aligned_face_primary:
        response["aligned_face"] = base64.b64encode(aligned_face_primary).decode('utf-8')
    
    return response

@router.get("/my-faces", response_model=Dict)
async def get_my_faces(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get information about registered faces for the current user, including images if available."""
    # Get face registrations with newest first
    embeddings = db.query(FaceEmbedding).filter(
        FaceEmbedding.user_id == current_user.id
    ).order_by(FaceEmbedding.created_at.desc()).all()
    
    result_faces = []
    for emb in embeddings:
        face_dict = {
            "id": emb.id,
            "created_at": emb.created_at,
            "device_id": emb.device_id,
            "confidence": emb.confidence_score
        }
        
        # Try to get the face image if it exists
        try:
            # Check if this embedding has an associated image
            if hasattr(emb, 'face_image') and emb.face_image and emb.face_image.image_data:
                face_dict["image"] = base64.b64encode(emb.face_image.image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error retrieving face image for embedding {emb.id}: {str(e)}")
        
        result_faces.append(face_dict)
    
    return {
        "count": len(embeddings),
        "faces": result_faces
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



@router.get("/my-faces", response_model=Dict)
async def get_user_face_registrations(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get face registrations for the current user."""
    # Get face registrations
    faces_query = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == current_user.id)
    faces = faces_query.order_by(FaceEmbedding.created_at.desc()).all()
    
    result_faces = []
    for face in faces:
        face_dict = {
            "id": face.id,
            "created_at": face.created_at,
            "device_id": face.device_id,
            "confidence_score": face.confidence_score
        }
        
        # Try to get the face image from the database or storage
        try:
            # Query the aligned face from the database (if you store it)
            face_image = db.query(FaceImage).filter(FaceImage.embedding_id == face.id).first()
            if face_image and face_image.image_data:
                face_dict["image"] = base64.b64encode(face_image.image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error retrieving face image: {str(e)}")
        
        result_faces.append(face_dict)
    
    return {"faces": result_faces}

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
    
    available_models = ["insightface"] 
    
    try:
        import deepface
        available_models.append("deepface")
        deepface_version = deepface.__version__
    except ImportError:
        deepface_version = None
    
    try:
        import face_recognition
        available_models.append("dlib")
        dlib_available = True
    except ImportError:
        dlib_available = False
    
    try:
        import insightface
        insightface_version = insightface.__version__
    except:
        insightface_version = "Unknown"
    
    return {
        "available_models": available_models,
        "model_versions": {
            "insightface": insightface_version,
            "deepface": deepface_version,
            "dlib": "Available" if dlib_available else "Not installed"
        },
        "anti_spoofing_supported": {
            "insightface": "Custom implementation",
            "deepface": "Yes" if "deepface" in available_models else "Not installed",
            "dlib": "Limited" if dlib_available else "Not installed"
        }
    }



