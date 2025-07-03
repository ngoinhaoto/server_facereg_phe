import os
import sys
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy_utils import database_exists, drop_database, create_database
from config.database import DATABASE_URL
from config.app import settings

def reset_db():
    """Reset the database by dropping and recreating it (no tables created here)"""
    print(f"Database URL: {DATABASE_URL}")
    
    # Drop the database if it exists
    if database_exists(DATABASE_URL):
        print(f"Dropping database: {settings.DB_NAME}")
        drop_database(DATABASE_URL)
        print(f"Database {settings.DB_NAME} dropped successfully!")
    
    # Create a fresh database
    print(f"Creating database: {settings.DB_NAME}")
    create_database(DATABASE_URL)
    print(f"Database {settings.DB_NAME} created successfully!")
    
    # Run Alembic migrations
    print("Running Alembic migrations (alembic upgrade head)...")
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Alembic migrations applied successfully!")
    else:
        print("Alembic migration failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print("Database reset complete!")
    return True

if __name__ == "__main__":
    try:
        import sqlalchemy_utils
    except ImportError:
        print("Installing sqlalchemy-utils...")
        os.system("pip install sqlalchemy-utils")
        from sqlalchemy_utils import database_exists, drop_database, create_database
    
    try:
        import psycopg2
    except ImportError:
        print("Installing psycopg2...")
        os.system("pip install psycopg2-binary")
    
    confirmation = input("⚠️  WARNING: This will DELETE all data in the database. Are you sure? (yes/no): ")
    if confirmation.lower() in ["yes", "y"]:
        if reset_db():
            # Optionally, seed the database after migrations
            seed_confirmation = input("Do you want to seed the database with sample data? (yes/no): ")
            if seed_confirmation.lower() in ["yes", "y"]:
                try:
                    from scripts.seed_db import create_sample_data
                    print("Seeding database with sample data...")
                    create_sample_data()
                    print("Database seeded successfully!")
                except Exception as e:
                    print(f"Error seeding database: {str(e)}")
        else:
            print("Database reset failed!")
    else:
        print("Database reset cancelled.")