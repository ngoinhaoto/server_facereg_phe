from typing import Dict, Optional, Tuple, Literal, ClassVar
import numpy as np
from sqlalchemy.orm import Session
import os
from utils.logging import logger

ModelType = Literal["insightface", "deepface"]

class FaceRecognitionBase:
    _instances: ClassVar[Dict[str, 'FaceRecognitionBase']] = {}
    
    @classmethod
    def get_instance(cls, model_type: ModelType = "deepface", det_size=(640, 640)):
        """Factory method to get or create an instance of the appropriate face recognition service"""
        from services.face_recognition.insightface_service import InsightFaceService
        from services.face_recognition.deepface_service import DeepFaceService
        
        key = f"{model_type}_{det_size[0]}_{det_size[1]}"
        if key not in cls._instances:
            if model_type == "insightface":
                cls._instances[key] = InsightFaceService(det_size)
            elif model_type == "deepface":
                cls._instances[key] = DeepFaceService()
            else:
                logger.error(f"Unsupported model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
        
        return cls._instances[key]
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        os.makedirs("./models", exist_ok=True)
    
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
        """Extract face embedding from an image"""
        raise NotImplementedError("Subclass must implement this method")
    
    def detect_spoofing(self, image_data: bytes) -> dict:
        """Detect if the image is a spoof (photo/screen) or real face"""
        raise NotImplementedError("Subclass must implement this method")
    
    def store_face_embedding(self, db: Session, user_id: int, embedding: np.ndarray, 
                 confidence: float, device_id: str = "web", model_type: str = None,
                 registration_group_id: str = None) -> int:
        """Store a face embedding in the database"""
        from models.database import FaceEmbedding
        import pickle
        
        try:
            model_to_use = model_type if model_type else self.model_type
            
            binary_embedding = pickle.dumps(embedding)
            
            db_embedding = FaceEmbedding(
                user_id=user_id,
                encrypted_embedding=binary_embedding,
                confidence_score=confidence,
                device_id=device_id,
                model_type=model_to_use,
                registration_group_id=registration_group_id
            )
            
            db.add(db_embedding)
            db.commit()
            db.refresh(db_embedding)
            logger.info(f"Stored face embedding for user {user_id} with model {model_to_use}")
            return db_embedding.id
        except Exception as e:
            logger.error(f"Error storing face embedding: {str(e)}")
            db.rollback()
            return 0
    
    def compare_face(self, embedding: np.ndarray, db: Session, user_id: int = None,
                 threshold: float = 0.5, model_type: str = None) -> Tuple[bool, Optional[int], float]:
        """Compare a face embedding with stored embeddings"""
        from models.database import FaceEmbedding
        import pickle
        
        try:
            # Use provided model_type or default to service's model_type
            model_to_use = model_type if model_type else self.model_type
            
            # Query embeddings
            query = db.query(FaceEmbedding)
            if user_id:
                query = query.filter(FaceEmbedding.user_id == user_id)
            
            # Filter by model type
            query = query.filter(FaceEmbedding.model_type == model_to_use)
            
            stored_embeddings = query.all()
            
            if not stored_embeddings:
                logger.warning(f"No stored embeddings found for comparison with model {model_to_use}")
                return False, None, 0.0
            
            best_match = None
            best_score = 0.0
            best_user_id = None
            
            for stored in stored_embeddings:
                # Deserialize the stored embedding
                try:
                    stored_embedding = pickle.loads(stored.encrypted_embedding)
                    
                    # Calculate cosine similarity
                    similarity = self.calculate_similarity(embedding, stored_embedding)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = stored
                        best_user_id = stored.user_id
                except Exception as e:
                    logger.error(f"Error deserializing embedding: {str(e)}")
                    continue
            
            # Check if the best match exceeds the threshold
            if best_score >= threshold:
                return True, best_user_id, best_score
            else:
                return False, None, best_score
                
        except Exception as e:
            logger.error(f"Error comparing face embeddings: {str(e)}")
            return False, None, 0.0
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # DeepFace uses L2 distance, InsightFace uses cosine similarity
        # We'll standardize on cosine similarity for both
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        dot_product = np.dot(embedding1, embedding2)
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range (InsightFace similarities can be slightly over 1)
        return min(max(float(similarity), 0.0), 1.0)
    
    def get_user_embeddings_count(self, db: Session, user_id: int) -> int:
        """Get the number of face embeddings stored for a user"""
        from models.database import FaceEmbedding
        return db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).count()
        
    def preprocess_image(self, file_data: bytes) -> bytes:
        """Preprocess image for better face detection"""
        import io
        from PIL import Image
        
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(file_data))
            
            # Convert to RGB (remove alpha channel if exists)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large while maintaining aspect ratio
            max_size = 1280
            if max(img.width, img.height) > max_size:
                ratio = max_size / max(img.width, img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert back to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            return img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return original data if processing fails
            return file_data