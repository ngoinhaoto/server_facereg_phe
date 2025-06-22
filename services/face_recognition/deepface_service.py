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
    
    def extract_face_embedding(self, image_data: bytes, check_spoofing=False) -> Tuple[Optional[np.ndarray], float, Optional[bytes], Optional[dict]]:
        """
        Extract embedding using DeepFace with optional anti-spoofing check
        
        Returns:
            Tuple containing:
            - embedding array (or None if not found)
            - confidence score
            - aligned face bytes (or None if not available)
            - anti-spoofing result dict (or None if check_spoofing=False)
        """
        temp_path = None
        try:
            logger.info(f"Starting DeepFace embedding extraction with anti_spoofing={check_spoofing}")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp.write(image_data)
                temp_path = temp.name
                logger.info(f"Image saved to temporary file: {temp_path}")
                
                detector_to_use = "opencv"  # More reliable than retinaface
                
                # First, if anti-spoofing is requested, check for spoofing
                spoof_result = None
                if check_spoofing:
                    try:
                        # Use extract_faces with anti_spoofing=True
                        face_objs = self.deepface.extract_faces(
                            img_path=temp_path,
                            detector_backend=detector_to_use,
                            enforce_detection=False,
                            align=True,
                            anti_spoofing=True
                        )
                        
                        logger.info(f"Anti-spoofing check completed, found {len(face_objs) if face_objs else 0} faces")
                        
                        # Process anti-spoofing results
                        is_spoof = True  # Default to spoof if no faces found
                        spoof_details = {"message": "No faces detected"}
                        spoof_score = 0.5 
                        
                        if face_objs and len(face_objs) > 0:
                            face_obj = face_objs[0]
                            is_real = face_obj.get("is_real", False)
                            
                            is_spoof = not is_real
                            
                            spoof_score = 0.0 if is_real else 0.8
                            
                            spoof_details = {
                                "message": "DeepFace anti-spoofing check",
                                "face_region": face_obj.get("facial_area", {}),
                                "confidence": face_obj.get("confidence", 0.0),
                                "all_faces_real": all(face.get("is_real", False) for face in face_objs)
                            }
                            
                            logger.info(f"DeepFace anti-spoofing result: is_real={is_real}")
                        
                        # Store result for return
                        spoof_result = {
                            "is_spoof": is_spoof,
                            "spoof_score": spoof_score,
                            "details": spoof_details,
                            "method": "deepface_native"
                        }
                        
                        # If it's a spoof and we don't need to continue with embedding, return early
                        if is_spoof:
                            logger.warning("Spoofing detected, skipping embedding extraction")
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            return None, 0.0, None, spoof_result
                            
                    except Exception as spoof_e:
                        logger.error(f"Error during anti-spoofing check: {str(spoof_e)}")
                        # Continue with embedding extraction even if spoofing check fails
                        spoof_result = {
                            "is_spoof": False,
                            "spoof_score": 0.0,
                            "details": {
                                "message": "Anti-spoofing check failed, defaulting to not a spoof",
                                "error": str(spoof_e)
                            },
                            "method": "deepface_fallback"
                        }
                
                # Now extract the embedding
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
                    fallback_embedding, fallback_confidence, fallback_face = self._fallback_extraction(temp_path)
                    return fallback_embedding, fallback_confidence, fallback_face, spoof_result
            
            # DeepFace returns a list of embedding objects
            if not embedding_obj or len(embedding_obj) == 0:
                logger.warning("No face embedding returned by DeepFace")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None, 0.0, None, spoof_result
            
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
            
            return embedding_array, confidence_score, aligned_face_bytes, spoof_result
            
        except Exception as e:
            logger.error(f"DeepFace face detection failed: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None, 0.0, None, {"is_spoof": False, "spoof_score": 0.0, "error": str(e), "method": "error"}
    
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
        """DeepFace-specific anti-spoofing implementation using the combined extraction method"""
        try:
            # Call the integrated method with check_spoofing=True
            _, _, _, spoof_result = self.extract_face_embedding(image_data, check_spoofing=True)
            
            # If we got a result, return it
            if spoof_result:
                return spoof_result
            
            # If no result was returned (shouldn't happen), return a default
            return {
                "is_spoof": False,
                "spoof_score": 0.0,
                "details": {"message": "No anti-spoofing result generated"},
                "method": "deepface_default"
            }
        except Exception as e:
            logger.error(f"Error in DeepFace anti-spoofing check: {str(e)}")
            
            return {
                "is_spoof": False,
                "spoof_score": 0.0,
                "details": {
                    "message": "Anti-spoofing check failed, defaulting to not a spoof",
                    "error": str(e)
                },
                "method": "deepface_fallback"
            }