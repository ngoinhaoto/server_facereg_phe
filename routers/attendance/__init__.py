from fastapi import APIRouter
from .check_in import router as check_in_router
from .face_management import router as face_management_router
from .reports import router as reports_router

# Create main router
router = APIRouter(prefix="/attendance", tags=["Attendance"])

# Include sub-routers
router.include_router(check_in_router)
router.include_router(face_management_router)
router.include_router(reports_router)

# Export the main router
__all__ = ["router"]