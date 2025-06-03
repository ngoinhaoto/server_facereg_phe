import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Union
import io
from sqlalchemy.orm import Session
from models.database import FaceEmbedding, User
import pickle
from utils.logging import logger
from PIL import Image

class FaceRecognitionService:
    
    def __init__(self, det_size=(640, 640)):
        import os
        os.makedirs("./models", exist_ok=True)
        
        # Initialize InsightFace
        self.app = FaceAnalysis(name="buffalo_l", root="./models")
        self.app.prepare(ctx_id=0, det_size=det_size)
        logger.info("InsightFace model loaded successfully")
        
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("Failed to decode image")
                return None, 0.0, None
                
            # RGB conversion (InsightFace expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.app.get(img)
            
            if not faces:
                logger.warning("No face detected in the image")
                return None, 0.0, None
            
            # Get the face with highest detection score
            face = max(faces, key=lambda x: x.det_score)
            
            # Get aligned face image for storage (optional)
            aligned_face_bytes = None
            try:
                # First try using the built-in method if available
                if hasattr(face, 'bbox_crop') and callable(face.bbox_crop):
                    aligned_face = face.bbox_crop(img)
                    if aligned_face is not None:
                        # Convert to bytes for storage
                        aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                        _, buf = cv2.imencode('.jpg', aligned_face_bgr)
                        aligned_face_bytes = buf.tobytes()
                else:
                    # Manual crop as fallback
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    # Add some margin (20%)
                    h, w = y2-y1, x2-x1
                    x1 = max(0, x1 - int(w*0.1))
                    y1 = max(0, y1 - int(h*0.1))
                    x2 = min(img.shape[1], x2 + int(w*0.1))
                    y2 = min(img.shape[0], y2 + int(h*0.1))
                    # Crop and convert
                    aligned_face = img[y1:y2, x1:x2]
                    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode('.jpg', aligned_face_bgr)
                    aligned_face_bytes = buf.tobytes()
            except Exception as e:
                logger.warning(f"Failed to crop face: {str(e)}")
                # Continue with the embedding even if cropping fails
        
            # Return embedding, confidence score, and aligned face
            return face.embedding, float(face.det_score), aligned_face_bytes
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None, 0.0, None
    
    def store_face_embedding(self, db: Session, user_id: int, embedding: np.ndarray, 
                             confidence: float, device_id: str = "web") -> bool:
        """
        Store a face embedding in the database
        
        Args:
            db: Database session
            staff_id
            embedding: Face embedding numpy array
            confidence: Confidence score
            device_id: Device identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize the numpy array
            binary_embedding = pickle.dumps(embedding)
            
            # Create new embedding record
            db_embedding = FaceEmbedding(
                user_id=user_id,
                encrypted_embedding=binary_embedding,
                confidence_score=confidence,
                device_id=device_id
            )
            
            db.add(db_embedding)
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing face embedding: {str(e)}")
            db.rollback()
            return False
    
    def compare_face(self, embedding: np.ndarray, db: Session, user_id: int = None,
                     threshold: float = 0.5) -> Tuple[bool, Optional[int], float]:
        """
        Compare a face embedding with stored embeddings
        
        Args:
            embedding: Face embedding to compare
            db: Database session
            user_id: Optional user ID to limit comparison to a specific user
            threshold: Similarity threshold (higher means stricter matching)
            
        Returns:
            Tuple of (match_found, user_id, similarity_score)
        """
        try:
            # Query embeddings
            query = db.query(FaceEmbedding)
            if user_id:
                query = query.filter(FaceEmbedding.user_id == user_id)
            
            stored_embeddings = query.all()
            
            if not stored_embeddings:
                logger.warning(f"No stored embeddings found for comparison")
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
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        dot_product = np.dot(embedding1, embedding2)
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range (InsightFace similarities can be slightly over 1)
        return min(max(float(similarity), 0.0), 1.0)
    
    def get_user_embeddings_count(self, db: Session, user_id: int) -> int:
        """
        Get the number of face embeddings stored for a user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Number of embeddings
        """
        return db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user_id).count()
        
    def preprocess_image(self, file_data: bytes) -> bytes:
        """
        Preprocess image for better face detection
        - Resize if too large
        - Adjust brightness/contrast if needed
        
        Args:
            file_data: Raw image bytes
            
        Returns:
            Processed image bytes
        """
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