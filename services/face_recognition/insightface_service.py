from typing import Optional, Tuple
import numpy as np
import cv2
from utils.logging import logger
from services.face_recognition.base import FaceRecognitionBase

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
    
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], float, Optional[bytes]]:
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
    
            return face.embedding, float(face.det_score), aligned_face_bytes
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None, 0.0, None
    
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
            
            # Crop face region
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
            
            # Ensure we have a valid prediction array
            if len(prediction) >= 2:
                real_score = float(prediction[0])
                spoof_score = float(prediction[1])
            else:
                # If prediction format is different, use first value as spoof score
                spoof_score = float(prediction[0])
                real_score = 1.0 - spoof_score
            
            # Set threshold (can be adjusted based on testing)
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
