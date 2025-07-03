import os
from pydantic_settings import BaseSettings  # Updated import

class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Recognition System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development-secret-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "password")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "fhe_facereg_db")
    
    # PHE Service URLs
    PHE_ONPREM_URL: str = os.getenv("PHE_ONPREM_URL", "http://localhost:8001")
    PHE_CLOUD_URL: str = os.getenv("PHE_CLOUD_URL", "http://localhost:8001")
    

    PHE_API_KEY: str = os.getenv("PHE_API_KEY", "your-secret-api-key")
    
    # Flag to enable/disable PHE
    ENABLE_PHE: bool = os.getenv("ENABLE_PHE", "True").lower() in ("true", "1", "t")
    
    # PHE settings
    PHE_KEYS_DIR: str = os.getenv("PHE_KEYS_DIR", "keys")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    
    # Database URL property
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Update to Pydantic v2 format
    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Allow extra fields in the environment
    }

settings = Settings()