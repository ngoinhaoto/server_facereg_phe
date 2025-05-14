import os
from dotenv import load_dotenv
from config.app import settings

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "facereg_db")

# SQLAlchemy connection string
DATABASE_URL = settings.DATABASE_URL