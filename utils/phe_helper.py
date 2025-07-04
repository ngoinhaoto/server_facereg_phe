import os
import requests
import logging
from config.app import settings
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def ensure_phe_public_key_exists():
    """
    Make sure the server has the public key from the PHE microservice.
    """
    os.makedirs("keys", exist_ok=True)
    public_key_path = "keys/public_key.txt"
    
    if not os.path.exists(public_key_path):
        try:
            logger.info("Public key not found. Fetching from PHE microservice...")
            try:
                phe_url = settings.PHE_ONPREM_URL
                response = requests.get(f"{phe_url}/public-key")
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Failed to connect to PHE microservice: {str(e)}")
                return False
            
            public_key = response.json().get("public_key")
            key_id = response.json().get("key_id", "default")
            
            if public_key:
                with open(public_key_path, "w") as f:
                    f.write(public_key)
                
                # Also store the key ID
                with open("keys/key_info.json", "w") as f:
                    json.dump({
                        "key_id": key_id,
                        "last_updated": datetime.now().isoformat()
                    }, f)
                    
                logger.info(f"Successfully fetched and saved PHE public key (ID: {key_id})")
            else:
                logger.error("Failed to fetch public key: Empty response")
                return False
        except Exception as e:
            logger.error(f"Error fetching PHE public key: {str(e)}")
            return False
    
    return True

def get_phe_instance():
    from lightphe import LightPHE
    
    if not ensure_phe_public_key_exists():
        logger.error("Could not initialize PHE: Public key not available")
        return None
    
    try:
        public_key_path = "keys/public_key.txt"
        phe_instance = LightPHE(algorithm_name="Paillier", precision=19, key_file=public_key_path)
        logger.info("Successfully initialized PHE with public key")
        return phe_instance
    except Exception as e:
        logger.error(f"Error initializing PHE with public key: {str(e)}")
        return None