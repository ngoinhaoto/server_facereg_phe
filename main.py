from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from security.auth import oauth2_scheme  # Import the OAuth2 scheme
from sqlalchemy.orm import Session
from database.db import get_db
import uvicorn
from routers import auth, users, classes, attendance  

# Create your FastAPI instance with security scheme
app = FastAPI(
    title="Face Recognition System",
    description="API for face recognition attendance system",
    version="0.1.0",
    openapi_tags=[
        {"name": "Authentication", "description": "Authentication operations"},
        {"name": "Users", "description": "User management operations"},
        {"name": "Classes", "description": "Class management operations"},
        {"name": "Attendance", "description": "Attendance tracking operations"},
    ],
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(classes.router)
app.include_router(attendance.router)

@app.get("/")
def read_root():
    return {"message": "Face Recognition API running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)