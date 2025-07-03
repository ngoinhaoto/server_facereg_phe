import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import Base
from database.db import engine
from sqlalchemy_utils import database_exists, create_database
from config.database import DATABASE_URL
from config.app import settings

def create_db():
    """Create the database and all tables"""
    print(f"Database URL: {DATABASE_URL}")
    
    # Create database if it doesn't exist
    if not database_exists(DATABASE_URL):
        print(f"Creating database: {settings.DB_NAME}")
        create_database(DATABASE_URL)
        print(f"Database {settings.DB_NAME} created successfully!")
    else:
        print(f"Database {settings.DB_NAME} already exists")
    
    # Create all tables defined in the models
    print("Creating tables from models...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
    
    return True

if __name__ == "__main__":
    try:
        import sqlalchemy_utils
    except ImportError:
        print("Installing sqlalchemy-utils...")
        os.system("pip install sqlalchemy-utils")
        from sqlalchemy_utils import database_exists, create_database
    
    if create_db():
        print("Database initialization complete!")
    else:
        print("Database initialization failed!")