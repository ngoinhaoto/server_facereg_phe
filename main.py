from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from security.auth import oauth2_scheme
from sqlalchemy.orm import Session
from database.db import get_db
import uvicorn
from routers import auth, users, classes, attendance
from routers.admin import dashboard
from config.app import settings

from utils.phe_helper import ensure_phe_public_key_exists, get_phe_instance
import logging

logger = logging.getLogger("phe-system")
logging.basicConfig(level=logging.INFO)

try:
    # Try both possible locations
    try:
        from routers.phe_face_management import router as phe_face_router
        phe_router_available = True
    except ImportError:
        from routers.attendance.phe_face_management import router as phe_face_router
        phe_router_available = True
except ImportError:
    phe_router_available = False

app = FastAPI(
    title="Face Recognition System with PHE",
    description="API for face recognition attendance system with Partial Homomorphic Encryption",
    version="0.2.0",
    openapi_tags=[
        {"name": "Authentication", "description": "Authentication operations"},
        {"name": "Users", "description": "User management operations"},
        {"name": "Classes", "description": "Class management operations"},
        {"name": "Attendance", "description": "Attendance tracking operations"},
        {"name": "Admin", "description": "Admin dashboard operations"},
        {"name": "PHE Face Management", "description": "Face operations with Partial Homomorphic Encryption"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(classes.router)
app.include_router(attendance.router)
app.include_router(
    dashboard.router,
    prefix="/admin",
    tags=["Admin"]
)

if settings.ENABLE_PHE and phe_router_available:
    app.include_router(phe_face_router)

phe_instance = None

@app.on_event("startup")
async def startup_event():
    global phe_instance
    
    # First check if the PHE public key exists
    if ensure_phe_public_key_exists():
        # Try to get the PHE instance
        phe_instance = get_phe_instance()
        if phe_instance:
            logger.info("PHE initialized successfully with public key")
        else:
            logger.warning("Failed to initialize PHE with public key")
    else:
        logger.warning("PHE public key does not exist")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Face Recognition API with PHE support",
        "docs": "/docs",
        "phe_enabled": settings.ENABLE_PHE
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "phe_enabled": settings.ENABLE_PHE}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)