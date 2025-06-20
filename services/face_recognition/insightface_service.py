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
        from insightface.app import FaceAnalysis
        
        self.app = FaceAnalysis(name="buffalo_l", root="./models")
        self.app.prepare(ctx_id=0, det_size=det_size)
        logger.info("InsightFace model loaded successfully")
    
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
                # Continue with the embedding even if cropping fails
    
            # Return embedding, confidence score, and aligned face
            return face.embedding, float(face.det_score), aligned_face_bytes
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            return None, 0.0, None
    
    def detect_spoofing(self, image_data: bytes) -> dict:
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