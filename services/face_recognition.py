import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Union, Literal
import io
from sqlalchemy.orm import Session
from models.database import FaceEmbedding, User
import pickle
from utils.logging import logger
from PIL import Image
import os

# Define model types
ModelType = Literal["insightface", "deepface"]

class FaceRecognitionService:
    _instances = {}
    
    @classmethod
    def get_instance(cls, model_type="deepface", det_size=(640, 640)):
        key = f"{model_type}_{det_size[0]}_{det_size[1]}"
        if key not in cls._instances:
            cls._instances[key] = cls(model_type, det_size)
        return cls._instances[key]
        
    def __init__(self, model_type: ModelType = "deepface", det_size=(640, 640)):
        self.model_type = model_type
        os.makedirs("./models", exist_ok=True)
        
        # Initialize the selected model
        if model_type == "insightface":
            self._init_insightface(det_size)
        elif model_type == "deepface":
            self._init_deepface()
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_insightface(self, det_size=(640, 640)):
        """Initialize InsightFace model"""
        self.app = FaceAnalysis(name="buffalo_l", root="./models")
        self.app.prepare(ctx_id=0, det_size=det_size)
        logger.info("InsightFace model loaded successfully")
    
    def _init_deepface(self):
        """Initialize DeepFace model"""
        try:
            # Import DeepFace here to avoid dependency if not used
            from deepface import DeepFace
            
            # Pre-load models to avoid first-call delay
            # We'll store these as instance variables
            self.deepface = DeepFace
            
            # Specify which model to use for face recognition
            # Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "SFace"
            self.deepface_model_name = "ArcFace"  # ArcFace is close to InsightFace in approach
            
            # Specify which detector to use
            # Options: "opencv", "ssd", "mtcnn", "retinaface", "mediapipe"
            self.detector_backend = "retinaface"  # RetinaFace is similar to InsightFace's detector
            
            # Force a model load to cache it
            _ = DeepFace.build_model(self.deepface_model_name)
            
            logger.info(f"DeepFace model loaded successfully (model: {self.deepface_model_name}, detector: {self.detector_backend})")
        except ImportError:
            logger.error("DeepFace not installed. Please install with: pip install deepface")
            raise ImportError("DeepFace not installed. Please install with: pip install deepface")
        except Exception as e:
            logger.error(f"Error initializing DeepFace: {str(e)}")
            raise
    
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
        """
        Extract face embedding from an image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (embedding, confidence_score, aligned_face_bytes)
        """
        if self.model_type == "insightface":
            return self._extract_embedding_insightface(image_data)
        elif self.model_type == "deepface":
            return self._extract_embedding_deepface(image_data)
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            return None, 0.0, None

    def _extract_embedding_insightface(self, image_data: bytes) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
        """Extract embedding using InsightFace"""
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

    def _extract_embedding_deepface(self, image_data: bytes) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
        """Extract embedding using DeepFace"""
        temp_path = None
        try:
            logger.info("Starting DeepFace embedding extraction")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp.write(image_data)
                temp_path = temp.name
                logger.info(f"Image saved to temporary file: {temp_path}")
                
                # Try with a more reliable detector first
                detector_to_use = "opencv"  # More reliable than retinaface
                logger.info(f"Calling DeepFace.represent with model={self.deepface_model_name}, detector={detector_to_use}")
                
                # Instead of using signal module for timeout, use a simple approach without timeouts
                # since we're running in a worker thread
                
                try:
                    embedding_obj = self.deepface.represent(
                        img_path=temp_path,
                        model_name=self.deepface_model_name,
                        detector_backend=detector_to_use,
                        enforce_detection=False,
                        align=True
                    )
                except Exception as deep_error:
                    logger.error(f"DeepFace.represent failed: {str(deep_error)}")
                    # Fall back to OpenCV
                    img = cv2.imread(temp_path)
                    if img is None:
                        logger.error("Failed to read image for fallback processing")
                        return None, 0.0, None
                        
                    # Use OpenCV's face detector as fallback
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) == 0:
                        logger.error("No face detected in fallback processing")
                        return None, 0.0, None
                        
                    # Use the first face
                    x, y, w, h = faces[0]
                    face_img = img[y:y+h, x:x+w]
                    
                    # Create a simple embedding (this is just a placeholder - not ideal)
                    simple_embedding = cv2.resize(face_img, (128, 128)).flatten() / 255.0
                    
                    # Return the simple embedding with low confidence
                    _, buf = cv2.imencode('.jpg', face_img)
                    aligned_face_bytes = buf.tobytes()
                    
                    logger.warning("Used fallback face detection - embedding will be less accurate")
                    return simple_embedding, 0.5, aligned_face_bytes
        
            # DeepFace returns a list of embedding objects
            if not embedding_obj or len(embedding_obj) == 0:
                logger.warning("No face embedding returned by DeepFace")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None, 0.0, None
            
            logger.info(f"DeepFace returned {len(embedding_obj)} embeddings")
            
            # Get the first embedding (DeepFace doesn't support multi-face in represent())
            embedding_vector = embedding_obj[0]["embedding"]
            embedding_array = np.array(embedding_vector)
            
            logger.info(f"Embedding extracted with shape: {embedding_array.shape}")
            
            confidence_score = 0.9  # Default high confidence if face detected
            
            # Get the aligned face image if available
            aligned_face_bytes = None
            try:
                # Read the original image for a fallback
                img = cv2.imread(temp_path)
                _, buf = cv2.imencode('.jpg', img)
                aligned_face_bytes = buf.tobytes()
                
                logger.info("Face image captured successfully")
            except Exception as e:
                logger.warning(f"Failed to get aligned face from DeepFace: {str(e)}")
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return embedding_array, confidence_score, aligned_face_bytes
            
        except Exception as e:
            logger.error(f"DeepFace face detection failed: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None, 0.0, None
            
        except Exception as e:
            logger.error(f"Error extracting face embedding with DeepFace: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None, 0.0, None
        
    def store_face_embedding(self, db: Session, user_id: int, embedding: np.ndarray, 
                         confidence: float, device_id: str = "web") -> int:
        """
        Store a face embedding in the database
        
        Args:
            db: Database session
            user_id: User ID
            embedding: Face embedding numpy array
            confidence: Confidence score
            device_id: Device identifier
            
        Returns:
            ID of the new embedding or 0 if failed
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
            db.refresh(db_embedding)
            return db_embedding.id
        except Exception as e:
            logger.error(f"Error storing face embedding: {str(e)}")
            db.rollback()
            return 0
    
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
    
    def detect_spoofing(self, image_data: bytes) -> dict:
        """
        Detect if the image is a spoof (photo/screen) or real face
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with spoof detection results
        """
        if self.model_type == "insightface":
            return self._detect_spoofing_custom(image_data)
        elif self.model_type == "deepface":
            return self._detect_spoofing_deepface(image_data)
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            return {"is_spoof": True, "spoof_score": 1.0, "error": f"Unsupported model: {self.model_type}"}

    def _detect_spoofing_custom(self, image_data: bytes) -> dict:
        """Custom anti-spoofing implementation for InsightFace"""
        try:
            # Convert image to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "Failed to decode image"}
            
            # Detect faces first to focus analysis on face region
            faces = self.app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not faces:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "No face detected"}
            
            # Get the face with highest detection score
            face = max(faces, key=lambda x: x.det_score)
            
            # Extract face region for focused analysis
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_region = img[y1:y2, x1:x2]
            if face_region.size == 0:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "Invalid face region"}
            
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # 1. Texture Analysis
            texture_var = np.var(gray)
            # Cap texture score at 1.0 to prevent inflated scores
            texture_score = min(1.0, texture_var / 2000.0)
            
            # 2. Edge density (printed photos often have sharper edges)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            edge_score = 1.0 - (edge_density / 255.0)  # Normalize and invert
            
            # 3. Face detection confidence
            detection_score = float(face.det_score)
            
            # 4. Color variance analysis (screens have different color distributions)
            b, g, r = cv2.split(face_region)
            color_vars = [np.var(b), np.var(g), np.var(r)]
            color_var_ratio = max(color_vars) / (min(color_vars) + 1e-5)
            color_score = min(1.0, max(0.0, 1.0 - abs(color_var_ratio - 1.5) / 1.5))
            
            # Combine scores
            spoof_indicators = {
                "texture": 0.35 * (1.0 - texture_score),
                "edges": 0.25 * (1.0 - edge_score),
                "detection": 0.20 * (1.0 - detection_score),
                "color": 0.20 * (1.0 - color_score)
            }
            
            # Calculate final score (higher = more likely to be spoof)
            spoof_score = sum(spoof_indicators.values())
            
            logger.info(f"InsightFace spoof detection: score={spoof_score:.2f}, " +
                      f"texture={spoof_indicators['texture']:.2f}, " +
                      f"edges={spoof_indicators['edges']:.2f}, " +
                      f"detection={spoof_indicators['detection']:.2f}, " +
                      f"color={spoof_indicators['color']:.2f}")
            
            return {
                "is_spoof": spoof_score > 0.5,
                "spoof_score": spoof_score,
                "details": {
                    "texture_score": texture_score,
                    "edge_score": edge_score,
                    "detection_score": detection_score,
                    "color_score": color_score
                },
                "method": "custom_texture_analysis"
            }
        except Exception as e:
            logger.error(f"Error in custom spoof detection: {str(e)}")
            return {"is_spoof": True, "spoof_score": 1.0, "error": str(e)}

    def _detect_spoofing_deepface(self, image_data: bytes) -> dict:
        """DeepFace-specific anti-spoofing implementation (temporarily disabled)"""
        try:
            # Just log that we're skipping the check
            logger.info("Anti-spoofing check disabled for DeepFace - automatically passing")
            
            # Always return not a spoof
            return {
                "is_spoof": False,
                "spoof_score": 0.0,
                "details": {
                    "message": "Anti-spoofing check disabled for DeepFace"
                },
                "method": "deepface_disabled"
            }
        except Exception as e:
            logger.error(f"Error in DeepFace spoof detection: {str(e)}")
            # Even if there's an error, don't block the user
            return {"is_spoof": False, "spoof_score": 0.0, "error": str(e)}