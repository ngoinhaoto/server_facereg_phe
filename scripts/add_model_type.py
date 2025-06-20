import os
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import SessionLocal
from models.database import FaceEmbedding
from config.face_recognition_config import face_recognition_config
from utils.logging import logger

def add_model_type_column():
    """Add model_type to existing face embeddings"""
    db = SessionLocal()
    try:
        # Get all embeddings without model_type (if column exists)
        try:
            embeddings = db.query(FaceEmbedding).filter(FaceEmbedding.model_type == None).all()
            print(f"Found {len(embeddings)} face embeddings without model_type")
        except:
            # If the column doesn't exist yet, SQLAlchemy will raise an error
            print("The model_type column does not exist yet. Run your database migrations first.")
            return
        
        # Default model from config
        default_model = face_recognition_config.DEFAULT_MODEL
        
        # Update each embedding
        for emb in embeddings:
            emb.model_type = default_model
        
        db.commit()
        print(f"Updated {len(embeddings)} embeddings to use model type {default_model}")
        
    except Exception as e:
        db.rollback()
        print(f"Error updating embeddings: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Adding model_type to existing face embeddings...")
    add_model_type_column()