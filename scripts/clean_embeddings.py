import os
import sys
import pickle
import numpy as np

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import SessionLocal
from models.database import FaceEmbedding
from utils.logging import logger

def clean_embeddings():
    """Remove corrupted face embeddings from the database"""
    db = SessionLocal()
    try:
        # Get all embeddings
        embeddings = db.query(FaceEmbedding).all()
        
        total = len(embeddings)
        removed = 0
        valid = 0
        
        print(f"Found {total} face embeddings in database")
        
        for embedding in embeddings:
            try:
                # Try to deserialize the embedding
                pickle.loads(embedding.encrypted_embedding)
                valid += 1
            except Exception as e:
                # If we can't deserialize, delete it
                print(f"Removing corrupted embedding ID {embedding.id} for user {embedding.user_id}: {str(e)}")
                db.delete(embedding)
                removed += 1
        
        db.commit()
        print(f"Completed: {removed} corrupted embeddings removed, {valid} valid embeddings kept")
        
    except Exception as e:
        db.rollback()
        print(f"Error cleaning embeddings: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Starting face embedding cleanup...")
    clean_embeddings()