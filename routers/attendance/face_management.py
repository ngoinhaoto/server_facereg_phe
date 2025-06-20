from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from sqlalchemy.orm import Session
from database.db import get_db
from services.face_recognition import FaceRecognitionService
from schemas.user import UserResponse
from security.auth import get_current_active_user
from models.database import FaceEmbedding, FaceImage
from starlette.concurrency import run_in_threadpool
import base64
from typing import Dict
from utils.logging import logger
from config.face_recognition_config import face_recognition_config

router = APIRouter()

@router.post("/register-face")
async def register_face(
    file: UploadFile = File(...),
    device_id: str = "web",
    model: str = None,
    store_both_models: bool = True,
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