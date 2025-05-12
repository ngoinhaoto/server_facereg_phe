import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import Base
from database.db import engine
from sqlalchemy_utils import database_exists, create_database
from config.database import DATABASE_URL

def init_db():
    # Create database if it doesn't exist
    if not database_exists(DATABASE_URL):
        print(f"Creating database: {DATABASE_URL}")
        create_database(DATABASE_URL)
    
    # Create all tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()