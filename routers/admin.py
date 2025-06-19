from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.db import get_db
from schemas.user import UserResponse
from security.auth import get_current_active_user
from typing import Dict, Any
from pydantic import BaseModel
from config.face_recognition_config import face_recognition_config, ModelType
from utils.logging import logger

router = APIRouter(prefix="/admin", tags=["Admin"])

class FaceRecognitionConfigUpdate(BaseModel):
    default_model: ModelType = None
    enable_antispoofing: bool = None
    enable_fallback: bool = None
    similarity_threshold: float = None

@router.get("/face-recognition-config")
async def get_face_recognition_config(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get current face recognition configuration"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can access face recognition configuration"
        )
    
    return {
        "default_model": face_recognition_config.DEFAULT_MODEL,
        "enable_antispoofing": face_recognition_config.ENABLE_ANTISPOOFING,
        "enable_fallback": face_recognition_config.ENABLE_FALLBACK,
        "similarity_threshold": face_recognition_config.SIMILARITY_THRESHOLD,
        "operation_overrides": {
            "register_face": face_recognition_config.REGISTER_FACE_MODEL,
            "check_in": face_recognition_config.CHECK_IN_MODEL
        }
    }

@router.post("/face-recognition-config")
async def update_face_recognition_config(
    config_update: FaceRecognitionConfigUpdate,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update face recognition configuration"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can update face recognition configuration"
        )
    
    # Update only the provided fields
    if config_update.default_model is not None:
        face_recognition_config.update_default_model(config_update.default_model)
    
    if config_update.enable_antispoofing is not None:
        face_recognition_config.ENABLE_ANTISPOOFING = config_update.enable_antispoofing
        logger.info(f"Anti-spoofing {'enabled' if config_update.enable_antispoofing else 'disabled'}")
    
    if config_update.enable_fallback is not None:
        face_recognition_config.ENABLE_FALLBACK = config_update.enable_fallback
        logger.info(f"Fallback {'enabled' if config_update.enable_fallback else 'disabled'}")
    
    if config_update.similarity_threshold is not None:
        face_recognition_config.SIMILARITY_THRESHOLD = config_update.similarity_threshold
        logger.info(f"Similarity threshold updated to {config_update.similarity_threshold}")
    
    return {"message": "Face recognition configuration updated successfully"}

@router.post("/face-recognition-config/operation-override")
async def set_operation_specific_model(
    operation: str,
    model: ModelType = None,  # None means use default
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Set model override for specific operation"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can update face recognition configuration"
        )
    
    # Validate operation
    valid_operations = ["register_face", "check_in"]
    if operation not in valid_operations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid operation. Valid operations are: {', '.join(valid_operations)}"
        )
    
    # Set the override
    override_attr = f"{operation.upper()}_MODEL"
    setattr(face_recognition_config, override_attr, model)
    
    if model is None:
        logger.info(f"Removed model override for {operation}, will use default")
        return {"message": f"Model override for {operation} removed, will use default"}
    else:
        logger.info(f"Set model override for {operation} to {model}")
        return {"message": f"Model override for {operation} set to {model}"}