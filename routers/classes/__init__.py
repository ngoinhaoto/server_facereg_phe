from fastapi import APIRouter
from .class_management import router as class_management_router
from .student_enrollment import router as student_enrollment_router
from .class_sessions import router as class_sessions_router

# Create main router
router = APIRouter(prefix="/classes", tags=["Classes"])

# Include sub-routers without prefix to maintain the same URLs
router.include_router(class_management_router)
router.include_router(student_enrollment_router)
router.include_router(class_sessions_router)