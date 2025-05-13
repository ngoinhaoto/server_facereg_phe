from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database.db import get_db
import uvicorn

app = FastAPI(title="Face Recognition System")

@app.get("/")
def read_root():
    return {"message": "Face Recognition API running"}

# Add your API routes here

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)