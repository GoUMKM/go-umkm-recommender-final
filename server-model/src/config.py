import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database configuration
    DB_HOST: str = os.getenv("DB_HOST", "go-umkm.c9e8y4qwgzqq.ap-southeast-1.rds.amazonaws.com")
    DB_PORT: int = os.getenv("DB_PORT", 49157)
    DB_USER: str = os.getenv("DB_USER", "remote_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "capstone_project")
    DB_NAME: str = os.getenv("DB_NAME", "go_umkm")
    
    # Model settings
    MODEL_VERSION: str = "1.0.0"
    TOP_K_RECOMMENDATIONS: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()