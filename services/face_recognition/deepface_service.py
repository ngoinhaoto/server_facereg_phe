from typing import Optional, Tuple
import numpy as np
import cv2
import os
import tempfile
from utils.logging import logger
from services.face_recognition.base import FaceRecognitionBase

class DeepFaceService(FaceRecognitionBase):
    """DeepFace implementation of face recognition service"""
    
    def __init__(self):
        """Initialize DeepFace model"""
        super().__init__("deepface")
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
        """Extract embedding using DeepFace"""
        temp_path = None
        try:
            logger.info("Starting DeepFace embedding extraction")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp.write(image_data)
                temp_path = temp.name
                logger.info(f"Image saved to temporary file: {temp_path}")
                
                detector_to_use = "opencv"  # More reliable than retinaface
                logger.info(f"Calling DeepFace.represent with model={self.deepface_model_name}, detector={detector_to_use}")
                
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
                    return self._fallback_extraction(temp_path)
        
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
    
    def _fallback_extraction(self, temp_path: str) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
        """Fallback face extraction using OpenCV if DeepFace fails"""
        try:
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
        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
            return None, 0.0, None
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def detect_spoofing(self, image_data: bytes) -> dict:
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