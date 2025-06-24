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
                spoof_result = self.detect_spoofing(img, face)  # Pass the image and face
    
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
    
    def detect_spoofing(self, image, face) -> dict:
        """
        Anti-spoofing using Face-AntiSpoofing ONNX model.
        Implementation follows the reference video_predict.py script exactly.
        
        Args:
            image: RGB image as numpy array
            face: Face object from InsightFace detector
            
        Returns:
            Dictionary with spoofing detection result
        """
        try:
            import os
            import numpy as np
            import cv2
            import onnxruntime as ort
            
            model_path = os.path.join("./models", "AntiSpoofing_bin_1.5_128.onnx")
            
            if not os.path.exists(model_path):
                logger.warning(f"Anti-spoofing model not found at: {model_path}")
                return self._fallback_spoofing_detection(face)
            
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
        
            # Get face bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Convert to format for increased_crop function
            bbox_xywh = (x1, y1, x2, y2)
            
            # Get increased crop around face - matching reference implementation
            face_img = self._increased_crop(image, bbox_xywh, bbox_inc=1.5)
            
            # Resize to model's expected input (128x128)
            face_img = cv2.resize(face_img, (128, 128))
            
            # Normalize to [0, 1] and convert to the right format (NCHW)
            face_tensor = face_img.astype(np.float32) / 255.0
            face_tensor = np.transpose(face_tensor, (2, 0, 1))  # HWC to CHW
            face_tensor = np.expand_dims(face_tensor, axis=0)  # Add batch dimension
            
            # Run inference
            outputs = self.antispoofing_session.run([self.antispoofing_output_name], 
                                                   {self.antispoofing_input_name: face_tensor})

            pred = outputs[0]
            score = float(pred[0][0])
            label = int(np.argmax(pred))
        
            threshold = face_recognition_config.ANTI_SPOOFING_THRESHOLD if hasattr(face_recognition_config, 'ANTI_SPOOFING_THRESHOLD') else 0.5
            
            if label == 0: 
                is_spoof = not (score > threshold)  # Only not spoof if score > threshold
                status = "REAL" if score > threshold else "UNKNOWN"
            else:  # Predicted as fake
                is_spoof = True
                status = "FAKE"
            
            logger.info(f"Anti-spoofing result: score={score:.2f}, label={label} (threshold: {threshold}) → {status}")
            
            result = {"is_spoof": is_spoof}
            
            if label == 1:  
                result["score"] = score
    
            return result
        except Exception as e:
            logger.error(f"Error in anti-spoofing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return self._fallback_spoofing_detection(face)

    def _increased_crop(self, img, bbox, bbox_inc=1.5):
        """
        Create an expanded crop around a face, exactly matching the reference implementation.
        """
        real_h, real_w = img.shape[:2]
        
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        l = max(w, h)
        
        xc, yc = x + w/2, y + h/2
        x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
        x1 = 0 if x < 0 else x 
        y1 = 0 if y < 0 else y
        x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
        y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
        
        img = img[y1:y2, x1:x2, :]
        img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img
    
    def _fallback_spoofing_detection(self, face) -> dict:
        """Fallback method for anti-spoofing detection (basic check)"""
        try:
            # Simple check - if we have a valid face with good confidence, assume it's not a spoof
            if face is not None and hasattr(face, 'det_score') and face.det_score > face_recognition_config.FACE_DETECTION_CONFIDENCE:
                return {"is_spoof": False}
            else:
                return {"is_spoof": True, "error": "Face detection confidence too low"}
        except Exception as e:
            logger.error(f"Error in fallback anti-spoofing detection: {str(e)}")
            return {"is_spoof": False, "error": str(e)}
