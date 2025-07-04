from typing import Dict, Optional, Tuple, Literal, ClassVar
import numpy as np
from sqlalchemy.orm import Session
import os
from utils.logging import logger

# Change this to only include "deepface"
ModelType = Literal["deepface"]

class FaceRecognitionBase:
    _instances: ClassVar[Dict[str, 'FaceRecognitionBase']] = {}
    
    @classmethod
    def get_instance(cls, model_type: ModelType = "deepface", det_size=(640, 640)):
        """Factory method to get or create an instance of the appropriate face recognition service"""
        from services.face_recognition.deepface_service import DeepFaceService
        
        key = f"{model_type}_{det_size[0]}_{det_size[1]}"
        if key not in cls._instances:
            if model_type == "deepface":
                cls._instances[key] = DeepFaceService()
            else:
                logger.error(f"Unsupported model type: {model_type}. Using DeepFace instead.")
                cls._instances[key] = DeepFaceService()
    
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
         registration_group_id: str = None, is_phe_encrypted: bool = False) -> int:
        from models.database import FaceEmbedding
        import pickle
        from main import phe_instance
        
        try:
            model_to_use = model_type if model_type else self.model_type
            embedding_size = len(embedding)
            
            # If the embedding is not already PHE encrypted, encrypt it now
            if not is_phe_encrypted and phe_instance:
                logger.info(f"Encrypting face embedding with PHE before storage")
                try:
                    # Check for negative values which can't be encrypted with PHE
                    if np.any(embedding < 0):
                        embedding = np.abs(embedding)
                        logger.warning("Converted negative embedding values to positive for PHE encryption")
                    
                    embedding_list = embedding.tolist()
                    
                    encrypted = phe_instance.encrypt(embedding_list)
                    binary_data = pickle.dumps(encrypted)
                    embedding_type = 'phe'
                except Exception as e:
                    logger.error(f"PHE encryption failed: {str(e)}. Falling back to plaintext storage.")
                    binary_data = pickle.dumps(embedding)
                    embedding_type = 'plaintext'
            else:
                # If it's already PHE encrypted or PHE is not available
                binary_data = pickle.dumps(embedding)
                embedding_type = 'phe' if is_phe_encrypted else 'plaintext'
            
            db_embedding = FaceEmbedding(
                user_id=user_id,
                embedding=binary_data,  # Store in the single field
                embedding_type=embedding_type,
                confidence_score=confidence,
                device_id=device_id,
                model_type=model_to_use,
                registration_group_id=registration_group_id,
                embedding_size=embedding_size
            )
            
            db.add(db_embedding)
            db.commit()
            db.refresh(db_embedding)
            logger.info(f"Stored {embedding_type} embedding for user {user_id} with model {model_to_use}")
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