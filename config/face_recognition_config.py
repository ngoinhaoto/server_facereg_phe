from typing import Literal, Optional
import os
from pydantic import BaseModel
from utils.logging import logger

# Update to only include "deepface"
ModelType = Literal["deepface"]

class FaceRecognitionConfig(BaseModel):
    """Configuration settings for face recognition services"""
    
    DEFAULT_MODEL: ModelType = "deepface"
    
    REGISTER_FACE_MODEL: Optional[ModelType] = None
    CHECK_IN_MODEL: Optional[ModelType] = None
    
    ENABLE_ANTISPOOFING: bool = True
    ENABLE_FALLBACK: bool = True
    
    ANTI_SPOOFING_THRESHOLD: float = 0.5 
    REGISTRATION_ANTI_SPOOFING_THRESHOLD: float = 0.7  
    ENABLE_REGISTRATION_ANTISPOOFING: bool = True  
    

    ENABLE_DUPLICATE_DETECTION: bool = False 
    DUPLICATE_DETECTION_THRESHOLD: float = 0.45 
    
    SIMILARITY_THRESHOLD: float = 0.5
    
    FACE_MIN_WIDTH_RATIO: float = 0.25  
    FACE_MIN_HEIGHT_RATIO: float = 0.25 
    FACE_MARGIN_RATIO: float = 0.05
    FACE_DETECTION_CONFIDENCE: float = 0.7 
    
    def get_model_for_operation(self, operation: str) -> ModelType:
        operation_upper = operation.upper()
        override_attr = f"{operation_upper}_MODEL"
        
        if hasattr(self, override_attr) and getattr(self, override_attr) is not None:
            model = getattr(self, override_attr)
            logger.info(f"Using operation-specific model for {operation}: {model}")
            return model
        
        logger.info(f"Using default model for {operation}: {self.DEFAULT_MODEL}")
        return self.DEFAULT_MODEL
    
    def get_anti_spoofing_threshold(self, is_registration: bool = False) -> float:
        """
        Get the appropriate anti-spoofing threshold based on operation
        
        Args:
            is_registration: Whether this is for registration (True) or check-in (False)
            
        Returns:
            The threshold value to use
        """
        if is_registration and self.ENABLE_REGISTRATION_ANTISPOOFING:
            return self.REGISTRATION_ANTI_SPOOFING_THRESHOLD
        return self.ANTI_SPOOFING_THRESHOLD
    
    def update_default_model(self, model: ModelType):
        """
        Update the default model used for all operations
        
        Args:
            model: The new default model
        """
        if model not in ["insightface", "deepface"]:
            raise ValueError(f"Invalid model: {model}")
        
        self.DEFAULT_MODEL = model
        logger.info(f"Default model updated to {model}")
    
    class Config:
        extra = "ignore"
        env_prefix = "FACE_RECOGNITION_"
        env_file = ".env"

# Create a singleton instance
face_recognition_config = FaceRecognitionConfig()