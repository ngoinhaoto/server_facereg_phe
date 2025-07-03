from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status, Request
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database.db import get_db
from schemas.user import UserResponse
from security.auth import get_current_active_user, get_current_user_or_none  # Add this
from models.database import FaceEmbedding, User, EmbeddingType  # Remove FaceImage
from services.phe_service import PHEService
import uuid
import base64
from typing import Optional, List
import logging
from utils.logging import logger
import requests
from fastapi import Security
from config.app import settings  
import pickle
import json



API_KEY = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(API_KEY)):
    if api_key != settings.PHE_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

router = APIRouter(prefix="/phe", tags=["PHE Face Management"])

phe_service = PHEService()

class EmbeddingData(BaseModel):
    embedding: List[float]
    
class EncryptedEmbeddingData(BaseModel):
    encrypted_embedding: str
    aligned_face: Optional[str] = None
    embedding_size: Optional[int] = 0
    user_id: Optional[int] = None 

@router.post("/register-face")
async def register_face_phe(
    file: UploadFile = File(...),
    device_id: str = "web",
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    try:
        image_data = await file.read()
        
        result = phe_service.extract_and_encrypt(image_data)
        
        encrypted_embedding = result.get("encrypted_embedding")
        aligned_face = result.get("aligned_face")
        embedding_size = result.get("embedding_size", 0)
        
        if not encrypted_embedding:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract and encrypt face embedding"
            )
        
        # Generate a unique ID for this registration
        registration_group_id = str(uuid.uuid4())
        
        db_embedding = FaceEmbedding(
            user_id=current_user.id,
            phe_embedding=encrypted_embedding,
            encrypted_embedding=None,  # No binary data for PHE embeddings
            embedding_type=EmbeddingType.PHE.value,
            confidence_score=0.9,
            device_id=device_id,
            model_type="vgg-face-phe",
            registration_group_id=registration_group_id,
            embedding_size=embedding_size
        )
        
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
        
        # Get total embeddings count
        embeddings_count = db.query(FaceEmbedding).filter(
            FaceEmbedding.user_id == current_user.id
        ).count()
        
        return {
            "message": "Face registered successfully with PHE",
            "embeddings_count": embeddings_count,
            "face_id": db_embedding.id,
            "embedding_type": EmbeddingType.PHE.value,
            "model_type": "vgg-face-phe"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering face with PHE: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering face: {str(e)}"
        )

@router.post("/verify-face")
async def verify_face_phe(
    file: UploadFile = File(...),
    user_id: Optional[int] = None,
    threshold: float = 0.6,
    db: Session = Depends(get_db)
):
    try:
        image_data = await file.read()
        
        embedding_result = phe_service.extract_embedding(image_data)
        plain_embedding = embedding_result.get("embedding")
        
        if not plain_embedding:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image"
            )
        
        query = db.query(FaceEmbedding, User).join(
            User, FaceEmbedding.user_id == User.id
        ).filter(
            FaceEmbedding.embedding_type == EmbeddingType.PHE.value
        )
        
        if user_id:
            query = query.filter(FaceEmbedding.user_id == user_id)
        
        stored_embeddings = query.all()
        
        if not stored_embeddings:
            return {
                "verified": False,
                "user_id": None,
                "user_name": None,
                "similarity": 0,
                "message": "No registered faces found"
            }
        
        best_match = None
        best_user = None
        best_similarity = 0
        
        for embedding, user in stored_embeddings:
            try:
                # Get the stored PHE embedding
                encrypted_embedding = embedding.phe_embedding
                
                # Compute similarity
                encrypted_similarity = phe_service.compute_similarity(plain_embedding, encrypted_embedding)
                
                # Decrypt the similarity score
                similarity = phe_service.decrypt_similarity(encrypted_similarity)
                
                # Update best match if this is better
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = embedding
                    best_user = user
            except Exception as e:
                logger.error(f"Error computing similarity for embedding: {str(e)}")
                continue
        
        if best_match is None:
            return {
                "verified": False,
                "user_id": None,
                "user_name": None,
                "similarity": 0,
                "message": "No valid comparisons could be made"
            }
        
        # Check if the similarity meets the threshold
        verified = best_similarity >= threshold
        
        return {
            "verified": verified,
            "user_id": best_user.id,
            "user_name": best_user.full_name or best_user.username,
            "similarity": best_similarity,
            "embedding_id": best_match.id,
            "model_type": best_match.model_type,
            "threshold": threshold
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying face with PHE")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error verifying face"
        )

@router.get("/status")
async def phe_status():
    """Check PHE services status"""
    try:
        # Check if PHE microservice is accessible
        onprem_status = "offline"
        cloud_status = "offline"
        
        try:
            # Check onprem service with timeout and disable SSL verification if needed
            onprem_response = requests.get(
                f"{settings.PHE_ONPREM_URL}/",
                timeout=2,  # Add a short timeout
                verify=False  # Disable SSL verification if using HTTPS
            )
            if onprem_response.status_code == 200:
                onprem_status = "online"
                logger.info(f"PHE onprem service is online: {onprem_response.json()}")
            else:
                logger.warning(f"PHE onprem service returned status code: {onprem_response.status_code}")
        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to PHE onprem service")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to PHE onprem service at {settings.PHE_ONPREM_URL}")
        except Exception as e:
            logger.error(f"Error checking onprem PHE service: {str(e)}")
        
        # Similar updates for cloud service check
        try:
            cloud_response = requests.get(
                f"{settings.PHE_CLOUD_URL}/",
                timeout=2,
                verify=False
            )
            if cloud_response.status_code == 200:
                cloud_status = "online"
                logger.info(f"PHE cloud service is online: {cloud_response.json()}")
            else:
                logger.warning(f"PHE cloud service returned status code: {cloud_response.status_code}")
        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to PHE cloud service")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to PHE cloud service at {settings.PHE_CLOUD_URL}")
        except Exception as e:
            logger.error(f"Error checking cloud PHE service: {str(e)}")
            
        return {
            "status": "PHE system status",
            "onprem_service": onprem_status,
            "cloud_service": cloud_status,
            "onprem_url": settings.PHE_ONPREM_URL,
            "cloud_url": settings.PHE_CLOUD_URL
        }
    except Exception as e:
        logger.error(f"Error checking PHE status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking PHE status: {str(e)}"
        )

@router.post("/store-encrypted-embedding")
async def store_encrypted_embedding(
    data: EncryptedEmbeddingData,
    request: Request,
    api_key: str = Security(API_KEY),
    db: Session = Depends(get_db),
    current_user: Optional[UserResponse] = Depends(get_current_user_or_none)
):
    """Store an encrypted embedding sent from PHE microservice"""
    try:
        user_id = None
        if current_user:
            user_id = current_user.id
        elif hasattr(data, 'user_id') and data.user_id:
            user_id = data.user_id
            
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
            
        registration_group_id = str(uuid.uuid4())
        
        decoded_data = base64.b64decode(data.encrypted_embedding)

        # Verify it's a valid EncryptedTensor by unpickling
        try:
            tensor = pickle.loads(decoded_data)
            from lightphe.models.Tensor import EncryptedTensor
            
            if not isinstance(tensor, EncryptedTensor):
                logger.warning(f"Received data is not an EncryptedTensor but {type(tensor).__name__}")
                # Try to recover if possible...
            else:
                logger.info(f"Successfully verified EncryptedTensor from microservice")
                # Store the decoded_data directly (the pickled bytes)
                final_embedding = decoded_data
                embedding_type = "encrypted_tensor"
                
        except Exception as e:
            logger.error(f"Failed to verify EncryptedTensor: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid embedding data: {str(e)}"
            )
        
        # Store in database
        embedding = FaceEmbedding(
            user_id=user_id,
            embedding=final_embedding,  # This is pickled EncryptedTensor bytes
            embedding_type=EmbeddingType.PHE.value,
            confidence_score=1.0,
            device_id=f"phe_microservice_{embedding_type}",
            model_type="deepface",
            registration_group_id=registration_group_id,
            embedding_size=data.embedding_size
        )
        
        db.add(embedding)
        db.commit()
        db.refresh(embedding)
        
        return {
            "message": "Encrypted embedding stored successfully",
            "embedding_id": embedding.id,
            "registration_group_id": registration_group_id,
            "embedding_type": embedding_type,
            "verified_as": "EncryptedTensor"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing encrypted embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing encrypted embedding: {str(e)}"
        )
    
    # Add a safe embedding inspector function at the top after imports
def safe_inspect_embedding(data):
    """Safely inspect embedding data without risking logging binary content"""
    if isinstance(data, bytes):
        return f"<binary data, length: {len(data)} bytes>"
    else:
        return f"<data of type: {type(data).__name__}>"

# First, add a detailed debugging helper at the top of the file
def detailed_debug_info(embedding_obj, user=None):
    """Generate detailed debug info about an embedding object"""
    result = {}
    
    # Basic embedding info
    result["embedding_id"] = embedding_obj.id if hasattr(embedding_obj, "id") else "unknown"
    result["embedding_type"] = embedding_obj.embedding_type if hasattr(embedding_obj, "embedding_type") else "unknown"
    result["model_type"] = embedding_obj.model_type if hasattr(embedding_obj, "model_type") else "unknown"
    result["device_id"] = embedding_obj.device_id if hasattr(embedding_obj, "device_id") else "unknown"
    
    # Embedding data info
    if hasattr(embedding_obj, "embedding") and embedding_obj.embedding:
        if isinstance(embedding_obj.embedding, bytes):
            result["embedding_format"] = "bytes"
            result["embedding_size"] = len(embedding_obj.embedding)
            # Try to peek at the data
            try:
                import pickle
                peek = pickle.loads(embedding_obj.embedding)
                result["deserialized_type"] = str(type(peek).__name__)
                from lightphe.models.Tensor import EncryptedTensor
                result["is_encrypted_tensor"] = isinstance(peek, EncryptedTensor)
            except:
                result["deserialization_error"] = "Failed to peek at binary data"
        else:
            result["embedding_format"] = str(type(embedding_obj.embedding).__name__)
            
    # User info if available
    if user:
        result["user_id"] = user.id
        result["username"] = user.username
    
    return result

@router.post("/verify-with-embedding")
async def verify_with_embedding(
    data: EmbeddingData,
    session_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Verify using an embedding extracted directly by the frontend with the microservice"""
    try:
        logger.info(f"Starting verification with session_id: {session_id}")
        plain_embedding = data.embedding
        
        query = db.query(FaceEmbedding, User).join(
            User, FaceEmbedding.user_id == User.id
        ).filter(
            FaceEmbedding.embedding_type == EmbeddingType.PHE.value
        )
        
        if session_id:
            logger.info(f"Verifying for specific session: {session_id}")
            from models.database import ClassSession, Class
            session_data = db.query(ClassSession).filter(ClassSession.id == session_id).first()
            if session_data:
                class_data = db.query(Class).filter(Class.id == session_data.class_id).first()
                if class_data and class_data.students:
                    student_ids = [student.id for student in class_data.students]
                    logger.info(f"Limiting verification to {len(student_ids)} students in class")
                    query = query.filter(User.id.in_(student_ids))
        
        stored_embeddings = query.all()
        
        if not stored_embeddings:
            return {
                "verified": False,
                "message": "No registered faces found",
                "results": [],
                "session_id": session_id,
                "embedding_count": 0
            }
        
        results = []
        errors = []
        
        for embedding_obj, user in stored_embeddings:
            try:
                # Try to deserialize the stored embedding
                try:
                    stored_embedding = pickle.loads(embedding_obj.embedding)
                    
                    from lightphe.models.Tensor import EncryptedTensor
                    if not isinstance(stored_embedding, EncryptedTensor):
                        logger.warning(f"Loaded embedding is not an EncryptedTensor but {type(stored_embedding).__name__}")
                        errors.append({
                            "user_id": user.id,
                            "error": f"Stored embedding is not an EncryptedTensor but {type(stored_embedding).__name__}"
                        })
                        continue
                    else:
                        logger.info(f"Successfully loaded EncryptedTensor for user {user.id}")
                except Exception as e:
                    logger.error(f"Error deserializing embedding for user {user.id}: {str(e)}")
                    errors.append({
                        "user_id": user.id,
                        "error": f"Error deserializing embedding: {str(e)}"
                    })
                    continue
                
                try:
                    encrypted_similarity = phe_service.compute_similarity(plain_embedding, stored_embedding)
                    
                    serialized_similarity = base64.b64encode(pickle.dumps(encrypted_similarity)).decode('utf-8')

                    results.append({
                        "user_id": user.id,
                        "username": user.username,
                        "full_name": user.full_name if hasattr(user, "full_name") else None,
                        "role": user.role if hasattr(user, "role") else None,
                        "encrypted_similarity": serialized_similarity,
                        "embedding_id": embedding_obj.id
                    })
                except Exception as e:
                    error_msg = f"Error computing similarity: {type(e).__name__}"
                    logger.error(f"Error computing similarity for user {user.id}: {str(e)}")
                    errors.append({
                        "user_id": user.id,
                        "error": error_msg
                    })
            except Exception as e:
                logger.error(f"Error processing embedding for user {user.id}: {str(e)}")
                errors.append({
                    "user_id": user.id,
                    "error": f"Error processing embedding: {str(e)}"
                })
        
        if len(results) == 0 and len(errors) > 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "message": "Failed to process any embeddings",
                    "errors": errors
                }
            )
        
        return {
            "results": results,
            "session_id": session_id,
            "embedding_count": len(results),
            "errors": errors if errors else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying with embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying with embedding"
        )

@router.post("/validate-embeddings", response_model=dict)
async def validate_embeddings(
    db: Session = Depends(get_db),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Validate and fix PHE embeddings in the database"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can validate embeddings"
        )
    
    embeddings = db.query(FaceEmbedding).filter(
        FaceEmbedding.embedding_type == EmbeddingType.PHE.value
    ).all()
    
    fixed = 0
    deleted = 0
    valid = 0
    errors = []
    
    for embedding in embeddings:
        try:
            # Try to deserialize the embedding
            from lightphe.models.Tensor import EncryptedTensor
            
            if embedding.embedding is None:
                logger.warning(f"Embedding {embedding.id} has null data - deleting")
                db.delete(embedding)
                deleted += 1
                continue
                
            try:
                loaded_data = pickle.loads(embedding.embedding)
                
                # Check if it's an EncryptedTensor
                if isinstance(loaded_data, EncryptedTensor):
                    valid += 1
                    continue
                    
                # If it's a string, try to decode and load
                if isinstance(loaded_data, str):
                    try:
                        decoded = base64.b64decode(loaded_data)
                        tensor = pickle.loads(decoded)
                        
                        if isinstance(tensor, EncryptedTensor):
                            # Fix the embedding
                            embedding.embedding = decoded
                            db.add(embedding)
                            fixed += 1
                            continue
                    except:
                        pass
                
                logger.warning(f"Embedding {embedding.id} is not an EncryptedTensor - deleting")
                db.delete(embedding)
                deleted += 1
                
            except Exception as e:
                logger.error(f"Error processing embedding {embedding.id}: {str(e)}")
                errors.append(f"Error on embedding {embedding.id}: {str(e)}")
                db.delete(embedding)
                deleted += 1
                
        except Exception as e:
            logger.error(f"Validation error for embedding {embedding.id}: {str(e)}")
            errors.append(f"Validation error for embedding {embedding.id}: {str(e)}")
    
    db.commit()
    
    return {
        "message": "Embedding validation completed",
        "total": len(embeddings),
        "valid": valid,
        "fixed": fixed,
        "deleted": deleted,
        "errors": errors if errors else None
    }


