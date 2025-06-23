from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from utils.logging import logger
from services.face_recognition.base import FaceRecognitionBase
from config.face_recognition_config import face_recognition_config

class InsightFaceService(FaceRecognitionBase):
    """InsightFace implementation of face recognition service"""
    
    def __init__(self, det_size=(640, 640)):
        """Initialize InsightFace model"""
        super().__init__("insightface")
        try:
            # Try to import with error handling
            from insightface.app import FaceAnalysis
            
            # Explicitly set providers for ONNX Runtime
            import os
            os.environ['ONNXRUNTIME_PROVIDERS_PATH'] = ''  # Reset providers path
            
            self.app = FaceAnalysis(name="buffalo_l", root="./models")
            self.app.prepare(ctx_id=0, det_size=det_size)
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize InsightFace: {str(e)}")
    
    def check_face_completeness(self, face, image) -> Tuple[bool, Optional[str]]:
        """
        Check if the face is complete (entire face is visible in the frame).
        
        Args:
            face: The face object from InsightFace
            image: The original image as numpy array
            
        Returns:
            Tuple of (is_complete, error_message)
        """
        try:
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Check face detection confidence
            if face.det_score < face_recognition_config.FACE_DETECTION_CONFIDENCE:
                return False, "Face detection confidence too low"
            
            # Get face bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Calculate face dimensions
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Check face size relative to image (ensures face is not too small)
            width_ratio = face_width / img_width
            height_ratio = face_height / img_height
            
            if width_ratio < face_recognition_config.FACE_MIN_WIDTH_RATIO:
                return False, "Face too small (width)"
                
            if height_ratio < face_recognition_config.FACE_MIN_HEIGHT_RATIO:
                return False, "Face too small (height)"
            
            # Check if face is too close to the edge of the frame
            margin_ratio = face_recognition_config.FACE_MARGIN_RATIO
            margin_x = img_width * margin_ratio
            margin_y = img_height * margin_ratio
            
            if x1 < margin_x or x2 > (img_width - margin_x) or y1 < margin_y or y2 > (img_height - margin_y):
                return False, "Face too close to image edge"
            
            # Check for key facial landmarks visibility
            # InsightFace provides landmarks as part of the face object
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = face.kps
                
                # Typical 5-point landmarks are: left eye, right eye, nose, left mouth corner, right mouth corner
                # Check if any landmark is outside the image or too close to the edge
                for landmark_idx, (x, y) in enumerate(landmarks):
                    if x < margin_x or x > (img_width - margin_x) or y < margin_y or y > (img_height - margin_y):
                        landmark_names = ["left eye", "right eye", "nose", "left mouth corner", "right mouth corner"]
                        landmark_name = landmark_names[landmark_idx] if landmark_idx < len(landmark_names) else f"landmark {landmark_idx}"
                        return False, f"Facial {landmark_name} not fully visible"
            else:
                logger.warning("Face landmarks not available for completeness check")
            
            # All checks passed
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking face completeness: {str(e)}")
            return False, f"Error checking face completeness: {str(e)}"
    
    def extract_face_embedding(self, image_data: bytes, check_spoofing=False) -> Tuple[Optional[np.ndarray], float, Optional[bytes], Optional[dict]]:
        """Extract embedding using InsightFace with optional anti-spoofing check"""
        try:
            spoof_result = None
            
            # Decode the image first (needed for both face detection and anti-spoofing)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("Failed to decode image")
                return None, 0.0, None, {"error": "Failed to decode image"}
                
            # RGB conversion (InsightFace expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces first - needed for both completeness check and anti-spoofing
            faces = self.app.get(img)
            
            if not faces:
                logger.warning("No face detected in the image")
                return None, 0.0, None, {"error": "No face detected in the image"}
            
            # Get the face with highest detection score
            face = max(faces, key=lambda x: x.det_score)
            
            # Check face completeness FIRST - no point doing anti-spoofing if face is incomplete
            is_complete, error_message = self.check_face_completeness(face, img)
            if not is_complete:
                logger.warning(f"Incomplete face detected: {error_message}")
                face_completeness_result = {
                    "is_spoof": False,  # Not spoofing, but still an error
                    "error": f"Incomplete face: {error_message}",
                    "incomplete_face": True
                }
                return None, 0.0, None, face_completeness_result
            
            # Only perform anti-spoofing after verifying the face is complete
            if check_spoofing:
                spoof_result = self.detect_spoofing(image_data)
                
                if spoof_result.get("is_spoof", False):
                    logger.warning("Spoofing detected in InsightFace, skipping embedding extraction")
                    return None, 0.0, None, spoof_result
        
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

            return face.embedding, float(face.det_score), aligned_face_bytes, spoof_result
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None, 0.0, None, {"error": str(e)}
    
    def detect_spoofing(self, image_data: bytes) -> dict:
        """Anti-spoofing using Face-AntiSpoofing ONNX model"""
        try:
            import os
            import numpy as np
            import cv2
            import onnxruntime as ort
            
            model_path = os.path.join("./models", "AntiSpoofing_bin_1.5_128.onnx")
            
            if not os.path.exists(model_path):
                logger.warning(f"Anti-spoofing model not found at: {model_path}")
                return self._fallback_spoofing_detection(image_data)
            
            if not hasattr(self, 'antispoofing_session'):
                available_providers = ort.get_available_providers()
                logger.info(f"Available ONNX Runtime providers: {available_providers}")
                
                self.antispoofing_session = ort.InferenceSession(
                    model_path,
                    providers=available_providers
                )
                
                self.antispoofing_input_name = self.antispoofing_session.get_inputs()[0].name
                self.antispoofing_output_name = self.antispoofing_session.get_outputs()[0].name
                logger.info("Anti-spoofing model loaded successfully")
            
            # Preprocess image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "Failed to decode image"}
            
            # Detect face first to focus analysis
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_img)
            if not faces:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "No face detected"}
            
            # Get the face with highest detection score
            face = max(faces, key=lambda x: x.det_score)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Add padding
            h, w = img.shape[:2]
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            face_img = img[y1:y2, x1:x2]
            
            # Resize to model's expected input (128x128) and convert to RGB
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (128, 128))
            
            # Normalize to [0, 1] and convert to the right format (NCHW)
            face_tensor = face_img.astype(np.float32) / 255.0
            face_tensor = np.transpose(face_tensor, (2, 0, 1))  # HWC to CHW
            face_tensor = np.expand_dims(face_tensor, axis=0)  # Add batch dimension
            
            # Run inference
            outputs = self.antispoofing_session.run([self.antispoofing_output_name], 
                                                   {self.antispoofing_input_name: face_tensor})
            
            # Get prediction - outputs should be probabilities for [real, fake]
            prediction = outputs[0][0]
            
            if len(prediction) >= 2:
                real_score = float(prediction[0])
                spoof_score = float(prediction[1])
            else:
                spoof_score = float(prediction[0])
                real_score = 1.0 - spoof_score
            
            threshold = 0.5
            
            logger.info(f"Anti-spoofing result: real={real_score:.2f}, spoof={spoof_score:.2f} (threshold: {threshold})")
            
            return {
                "is_spoof": spoof_score > threshold,
                "spoof_score": spoof_score,
                "details": {
                    "real_score": real_score,
                    "detection_confidence": float(face.det_score)
                },
                "method": "face_antispoofing"
            }
        except Exception as e:
            logger.error(f"Error in anti-spoofing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fall back to simple detection
            return self._fallback_spoofing_detection(image_data)
    
    def _fallback_spoofing_detection(self, image_data: bytes) -> dict:
        """Fallback method for anti-spoofing detection (basic check)"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "Failed to decode image"}
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_img)
            if not faces:
                return {"is_spoof": True, "spoof_score": 1.0, "error": "No face detected"}
            
            return {"is_spoof": False, "spoof_score": 0.0}
        except Exception as e:
            logger.error(f"Error in fallback anti-spoofing detection: {str(e)}")
            return {"is_spoof": False, "spoof_score": 0.0, "error": str(e)}
