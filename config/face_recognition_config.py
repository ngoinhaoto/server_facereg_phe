from typing import Literal, Optional
import os
from pydantic import BaseModel
from utils.logging import logger

ModelType = Literal["insightface", "deepface"]

class FaceRecognitionConfig(BaseModel):
    """Configuration settings for face recognition services"""
    
    DEFAULT_MODEL: ModelType = "deepface"
    
    REGISTER_FACE_MODEL: Optional[ModelType] = None
    CHECK_IN_MODEL: Optional[ModelType] = None
    
    ENABLE_ANTISPOOFING: bool = True
    ENABLE_FALLBACK: bool = True
    
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