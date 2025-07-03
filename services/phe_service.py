from typing import List, Dict, Any, Optional
import requests
import ast
from utils.logging import logger
from config.app import settings

class PHEService:
    """
    Service for interacting with PHE microservices.
    
    The server only has access to the public key and cannot decrypt data.
    It communicates with the PHE microservice for operations requiring the private key.
    """
    
    def __init__(self):
        # URLs for PHE microservices
        self.onprem_url = settings.PHE_ONPREM_URL  # Client-side service with private key
        self.cloud_url = settings.PHE_CLOUD_URL    # Could be same as onprem for testing
    
    def extract_embedding(self, image_data: bytes) -> Dict[str, Any]:
        """Extract face embedding using the client-side PHE microservice"""
        try:
            files = {'file': ('image.jpg', image_data)}
            response = requests.post(
                f"{self.onprem_url}/extract-embedding",
                files=files
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            raise
    
    def encrypt_embedding(self, embedding: List[float]) -> str:
        """Encrypt a face embedding using the client-side PHE microservice"""
        try:
            payload = {"embedding": embedding}
            response = requests.post(
                f"{self.onprem_url}/encrypt",
                json=payload
            )
            response.raise_for_status()
            return response.json()["encrypted_embedding"]
        except Exception as e:
            logger.error(f"Error encrypting embedding: {str(e)}")
            raise
    
    def compute_similarity(self, plain_embedding: List[float], encrypted_embedding) -> str:
        """
        Compute similarity between plain and encrypted embeddings using the @ operator
        
        This is done on the server side (exactly as in your notebook),
        but the result remains encrypted and can only be decrypted by the client.
        """
        try:
            from lightphe.models.Tensor import EncryptedTensor
            
            if isinstance(encrypted_embedding, EncryptedTensor):
                # Already an EncryptedTensor object, use it directly
                enc_embedding = encrypted_embedding
            
            # Use @ operator (matrix multiplication) for similarity computation
            encrypted_similarity = enc_embedding @ plain_embedding
            
            return encrypted_similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
    
    def extract_and_encrypt(self, image_data: bytes) -> Dict[str, Any]:
        """Extract and encrypt in one operation using the client-side PHE microservice"""
        try:
            files = {'file': ('image.jpg', image_data)}
            response = requests.post(
                f"{self.onprem_url}/extract-and-encrypt",
                files=files
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in extract and encrypt: {str(e)}")
            raise