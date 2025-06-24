from typing import List, Dict, Optional, Tuple
import numpy as np
from utils.logging import logger
from config.face_recognition_config import face_recognition_config
from models.database import User, FaceEmbedding

class DuplicateFaceDetector:
    """Service to detect duplicate faces across users"""
    
    @staticmethod
    async def check_for_duplicates(embedding: np.ndarray, current_user_id: Optional[int] = None) -> Tuple[bool, Optional[Dict]]:
        if not face_recognition_config.ENABLE_DUPLICATE_DETECTION:
            return False, None
            
        try:
            import pickle
            from sqlalchemy import select, join
            from database.db import get_db, SessionLocal
            
            highest_similarity = 0.0
            most_similar_user = None
            most_similar_embedding = None
            duplicate_threshold = face_recognition_config.DUPLICATE_DETECTION_THRESHOLD
            
            # Create a database session directly without using async for
            db = SessionLocal()
            try:
                # Query for all face embeddings except those belonging to current user
                query = select(FaceEmbedding, User).join(User, FaceEmbedding.user_id == User.id)
                if current_user_id is not None:
                    query = query.where(FaceEmbedding.user_id != current_user_id)
                    
                result = db.execute(query)
                embeddings_with_users = result.all()
                
                for face_embedding, user in embeddings_with_users:
                    if face_embedding.encrypted_embedding is not None:
                        try:
                            # Use pickle.loads to unpickle the embedding
                            stored_embedding = pickle.loads(face_embedding.encrypted_embedding)
                            
                            # Calculate cosine similarity
                            similarity = np.dot(embedding, stored_embedding) / (
                                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                            )
                            
                            logger.debug(f"Similarity with user {user.username}: {similarity:.4f}")
                            
                            # Track highest similarity
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                most_similar_user = user
                                most_similar_embedding = face_embedding
                            
                            # Check if this exceeds the duplicate threshold
                            if similarity >= duplicate_threshold:
                                logger.warning(f"Duplicate face detected! Similarity: {similarity:.4f}, User ID: {user.id}, Username: {user.username}")
                                return True, {
                                    "is_duplicate": True,
                                    "similarity": float(similarity),
                                    "duplicate_user_id": user.id,
                                    "duplicate_user_name": user.full_name if user.full_name else user.username,
                                    "embedding_id": face_embedding.id,
                                    "model_type": face_embedding.model_type
                                }
                        except Exception as inner_e:
                            logger.error(f"Error processing embedding {face_embedding.id}: {str(inner_e)}")
                            continue
            finally:
                db.close()
            
            if most_similar_user and highest_similarity > (duplicate_threshold * 0.8):
                logger.info(f"Similar face detected (but below threshold). Similarity: {highest_similarity:.4f}, User ID: {most_similar_user.id}")
                return False, {
                    "is_duplicate": False,
                    "highest_similarity": float(highest_similarity),
                    "most_similar_user_id": most_similar_user.id,
                    "most_similar_user_name": most_similar_user.full_name if most_similar_user.full_name else most_similar_user.username,
                    "embedding_id": most_similar_embedding.id if most_similar_embedding else None,
                    "model_type": most_similar_embedding.model_type if most_similar_embedding else None
                }
                
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking for duplicate faces: {str(e)}")
            return False, {"error": str(e)}