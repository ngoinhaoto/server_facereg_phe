from typing import List, Optional, Tuple
import numpy as np
import cv2
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import functools
from utils.logging import logger
from services.face_recognition.base import FaceRecognitionBase
from config.face_recognition_config import face_recognition_config

def process_batch_embeddings(self, image_data_list: List[bytes]) -> List[Tuple[Optional[np.ndarray], float, Optional[bytes]]]:
    results = []
    
    if not image_data_list:
        return results
    
    max_workers = 8 #since mac m1 has 8 cores

    # if self.device and 'cuda' in str(self.device):
    #     max_workers = ...
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.extract_face_embedding, img_data, False) 
                      for img_data in image_data_list]
            
            # Collect results as they complete
            for future in futures:
                embedding, confidence, face_img, _ = future.result()
                results.append((embedding, confidence, face_img))
    
    except Exception as e:
        logger.error(f"Error in GPU-optimized batch processing: {str(e)}")
    
    self._manage_memory()
    
    return results

class DeepFaceService(FaceRecognitionBase):
    """DeepFace implementation of face recognition service"""
    
    def __init__(self):
        """Initialize DeepFace model with GPU support for both M1 and CUDA"""
        super().__init__("deepface")
        try:
            from deepface import DeepFace
            
            self.device = self._setup_gpu_acceleration()
            
            self.deepface = DeepFace
            
            # Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace", "SFace"
            self.deepface_model_name = "Facenet512"

            self.detector_backend = "yunet" #other options: "opencv", "ssd", "mtcnn", "retinaface", "centerface"
            
            # Force model load with optimizations for the detected hardware
            self._load_optimized_model()
            
            logger.info(f"DeepFace model loaded successfully (model: {self.deepface_model_name}, detector: {self.detector_backend}, device: {self.device})")
        except ImportError:
            logger.error("DeepFace not installed. Please install with: pip install deepface")
            raise ImportError("DeepFace not installed. Please install with: pip install deepface")
        except Exception as e:
            logger.error(f"Error initializing DeepFace: {str(e)}")
            raise
    
    def _setup_gpu_acceleration(self):
        """Set up the best available GPU acceleration (CUDA or Metal)"""
        device = None
        try:
            import torch
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
                cuda_device = torch.cuda.get_device_properties(0)
                logger.info(f"CUDA acceleration enabled: {cuda_device.name} with {cuda_device.total_memory/1024**3:.1f}GB memory")
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Apple Metal Performance Shaders (MPS) acceleration enabled")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            else:
                device = torch.device("cpu")
                logger.info("GPU acceleration not available, using CPU")
        except ImportError:
            logger.info("PyTorch not available, using CPU acceleration")
        
        return device

    def _load_optimized_model(self):
        try:
            _ = self.deepface.build_model(self.deepface_model_name)
            
            if self.device:
                if 'cuda' in str(self.device):
                    os.environ["OMP_NUM_THREADS"] = "8" 
                    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
                    
                    # Try to enable TensorRT if available (for NVIDIA GPUs)
                    try:
                        import tensorflow as tf
                        if hasattr(tf, 'config') and hasattr(tf.config, 'optimizer'):
                            tf.config.optimizer.set_jit(True)
                            logger.info("TensorFlow JIT optimization enabled for CUDA")
                    except:
                        pass
                        
                elif 'mps' in str(self.device):
                    os.environ["OMP_NUM_THREADS"] = "8" 
                    os.environ["MKL_NUM_THREADS"] = "8"
                    
                    try:
                        import subprocess
                        result = subprocess.run(['pmset', '-g', 'batt'], 
                                              capture_output=True, text=True)
                        if 'discharging' in result.stdout:
                            logger.info("Running on battery, enabling power-saving mode")
                            os.environ["PYTORCH_MPS_LOW_PRECISION"] = "1"
                    except Exception as e:
                        logger.debug(f"Could not check battery status: {e}")
                else:
                    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
                    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 4)
        except Exception as e:
            logger.warning(f"Error during optimized model loading: {e}")
    
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
                
                detector_to_use = "opencv"
                
                img = cv2.imread(temp_path)
                
                if img is None:
                    logger.error("Failed to read image")
                    return None, 0.0, None, {"error": "Failed to read image"}
                
                try:
                    face_objs = self.deepface.extract_faces(
                        img_path=temp_path,
                        detector_backend=detector_to_use,
                        enforce_detection=False,
                        align=True
                    )
                    
                    if not face_objs or len(face_objs) == 0:
                        logger.warning("No faces detected in image")
                        return None, 0.0, None, {"error": "No face detected in the image"}
                    
                    is_complete, error_message = self.check_face_completeness(face_objs[0], img)
                    if not is_complete:
                        logger.warning(f"Incomplete face detected: {error_message}")
                        face_completeness_result = {
                            "is_spoof": False,  # Not spoofing, but still an error
                            "error": f"Incomplete face: {error_message}",
                            "incomplete_face": True
                        }
                        return None, 0.0, None, face_completeness_result
                    
                    logger.info("Face completeness check passed")
                    
                except Exception as face_e:
                    logger.error(f"Error during face detection or completeness check: {str(face_e)}")
                    return None, 0.0, None, {"error": f"Error detecting face: {str(face_e)}"}
                
                spoof_result = None
                if check_spoofing:
                    try:
                        anti_spoof_faces = self.deepface.extract_faces(
                            img_path=temp_path,
                            detector_backend=detector_to_use,
                            enforce_detection=False,
                            align=True,
                            anti_spoofing=True
                        )
                        
                        logger.info(f"Anti-spoofing check completed, found {len(anti_spoof_faces) if anti_spoof_faces else 0} faces")
                        
                        is_spoof = True  
                        spoof_details = {"message": "No faces detected"}
                        spoof_score = 0.5 
                        
                        if anti_spoof_faces and len(anti_spoof_faces) > 0:
                            face_obj = anti_spoof_faces[0]
                            is_real = face_obj.get("is_real", False)
                            
                            is_spoof = not is_real
                            
                            spoof_score = 0.0 if is_real else 0.8
                            
                            spoof_details = {
                                "message": "DeepFace anti-spoofing check",
                                "face_region": face_obj.get("facial_area", {}),
                                "confidence": face_obj.get("confidence", 0.0),
                                "all_faces_real": all(face.get("is_real", False) for face in anti_spoof_faces)
                            }
                            
                            logger.info(f"DeepFace anti-spoofing result: is_real={is_real}")
                        
                        spoof_result = {
                            "is_spoof": is_spoof,
                            "spoof_score": spoof_score,
                            "details": spoof_details,
                            "method": "deepface_native"
                        }
                        
                        if is_spoof:
                            logger.warning("Spoofing detected, skipping embedding extraction")
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            return None, 0.0, None, spoof_result
                            
                    except Exception as spoof_e:
                        logger.error(f"Error during anti-spoofing check: {str(spoof_e)}")

                        spoof_result = {
                            "is_spoof": False,
                            "spoof_score": 0.0,
                            "details": {
                                "message": "Anti-spoofing check failed, defaulting to not a spoof",
                                "error": str(spoof_e)
                            },
                            "method": "deepface_fallback"
                        }
            
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
            
            if not embedding_obj or len(embedding_obj) == 0:
                logger.warning("No face embedding returned by DeepFace")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None, 0.0, None, spoof_result
            
            logger.info(f"DeepFace returned {len(embedding_obj)} embeddings")
            
            embedding_vector = embedding_obj[0]["embedding"]
            embedding_array = np.array(embedding_vector)
            
            logger.info(f"Embedding extracted with shape: {embedding_array.shape}")
            
            confidence_score = face_objs[0].get("confidence", 0.9)
            
            aligned_face_bytes = None
            try:
                facial_area = face_objs[0]["facial_area"]
                x = facial_area.get("x", 0)
                y = facial_area.get("y", 0)
                w = facial_area.get("w", 0)
                h = facial_area.get("h", 0)
                
                img = cv2.imread(temp_path)
                
                img_h, img_w = img.shape[:2]
                
                x = max(0, x)
                y = max(0, y)
                w = min(img_w - x, w)
                h = min(img_h - y, h)
                
                face_img = img[y:y+h, x:x+w]
                logger.info(f"DeepFace: using exact face region from detector ({face_img.shape[:2]})")
                
                if face_img.size == 0:
                    logger.warning("Extracted face region is empty, using whole image")
                    face_img = img
                
                _, buf = cv2.imencode('.jpg', face_img)
                aligned_face_bytes = buf.tobytes()
                
                logger.info(f"Face image extracted, size: {len(aligned_face_bytes)} bytes")
            except Exception as e:
                logger.warning(f"Failed to extract face: {str(e)}")
                try:
                    img = cv2.imread(temp_path)
                    _, buf = cv2.imencode('.jpg', img)
                    aligned_face_bytes = buf.tobytes()
                    logger.info("Using whole image as fallback")
                except Exception as fallback_e:
                    logger.error(f"Fallback image extraction failed: {str(fallback_e)}")
            
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
            img = cv2.imread(temp_path)
            if img is None:
                logger.error("Failed to read image for fallback processing")
                return None, 0.0, None
                
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                logger.error("No face detected in fallback processing")
                return None, 0.0, None
                
            x, y, w, h = faces[0]
            
            img_h, img_w = img.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(img_w - x, w)
            h = min(img_h - y, h)
            
            # Extract face region directly
            face_img = img[y:y+h, x:x+w]
            
            # Create a simple embedding
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
    
    def _manage_memory(self):
        """Optimize memory usage based on detected hardware"""
        try:
            # Import memory management utilities
            import gc
            gc.collect()
            
            # Device-specific memory management
            if self.device:
                if 'cuda' in str(self.device):
                    # CUDA-specific memory management
                    try:
                        import torch
                        if torch.cuda.is_available():
                            # Clear CUDA cache
                            torch.cuda.empty_cache()
                            
                            # Log GPU memory usage if debug level
                            if logger.level <= 10:  # DEBUG level
                                free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                                total_mem = torch.cuda.get_device_properties(0).total_memory
                                logger.debug(f"CUDA memory: {free_mem/1024**2:.1f}MB free / {total_mem/1024**2:.1f}MB total")
                    except Exception as e:
                        logger.debug(f"Error in CUDA memory management: {e}")
                        
                elif 'mps' in str(self.device):
                    # M1-specific memory management
                    try:
                        import torch
                        
                        # For MPS, we force a garbage collection
                        gc.collect()
                        
                        # Try to get memory info for M1
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_info = process.memory_info()
                            logger.debug(f"Process memory: {memory_info.rss/1024**2:.1f}MB")
                        except:
                            pass
                    except Exception as e:
                        logger.debug(f"Error in M1 memory management: {e}")
        except Exception as e:
            logger.error(f"Error in memory management: {str(e)}")
    
    def check_face_completeness(self, face_obj, img=None) -> Tuple[bool, Optional[str]]:
        """
        Check if the face is complete (entire face is visible in the frame).
        
        Args:
            face_obj: The face object from DeepFace extract_faces
            img: Original image (numpy array) if available
            
       """
        try:
            if not face_obj:
                return False, "No face detected"
                
            # If it's a list, use the first face
            if isinstance(face_obj, list) and len(face_obj) > 0:
                face_obj = face_obj[0]
            
            # Get image dimensions if available
            img_width, img_height = 0, 0
            if img is not None:
                img_height, img_width = img.shape[:2]
            
            # Get face region (facial_area) from DeepFace
            if "facial_area" not in face_obj:
                return False, "No facial area information available"
                
            facial_area = face_obj["facial_area"]
            
            # Get face coordinates
            x = facial_area.get("x", 0)
            y = facial_area.get("y", 0)
            w = facial_area.get("w", 0)
            h = facial_area.get("h", 0)
            
            # If we couldn't determine image dimensions earlier, try to infer from face
            if img_width == 0 and "img" in face_obj:
                face_img = face_obj["img"]
                if face_img is not None:
                    img_height, img_width = face_img.shape[:2]
                    # Since this is the face crop, we need to adjust based on known coordinates
                    img_width = max(img_width, x + w)
                    img_height = max(img_height, y + h)
            
            # If we still don't have image dimensions, use detection confidence only
            if img_width == 0:
                if "confidence" in face_obj and face_obj["confidence"] < face_recognition_config.FACE_DETECTION_CONFIDENCE:
                    return False, "Face detection confidence too low"
                return True, None
            
            width_ratio = w / img_width
            height_ratio = h / img_height
            
            if width_ratio < face_recognition_config.FACE_MIN_WIDTH_RATIO:
                return False, "Face too small (width)"
                
            if height_ratio < face_recognition_config.FACE_MIN_HEIGHT_RATIO:
                return False, "Face too small (height)"
            
            # Check if face is too close to the edge
            margin_ratio = face_recognition_config.FACE_MARGIN_RATIO
            margin_x = img_width * margin_ratio
            margin_y = img_height * margin_ratio
            
            if x < margin_x or (x + w) > (img_width - margin_x) or y < margin_y or (y + h) > (img_height - margin_y):
                return False, "Face too close to image edge"
            
            if "confidence" in face_obj and face_obj["confidence"] < face_recognition_config.FACE_DETECTION_CONFIDENCE:
                return False, "Face detection confidence too low"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error checking face completeness in DeepFace: {str(e)}")
            return False, f"Error checking face completeness: {str(e)}"