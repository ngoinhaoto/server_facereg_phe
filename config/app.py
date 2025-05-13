import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Recognition System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development-secret-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    
    CONCRETML_MODEL_PATH: str = os.getenv("CONCRETML_MODEL_PATH", "./models/face_model")
    
    class Config:
        env_file = ".env"

settings = Settings()