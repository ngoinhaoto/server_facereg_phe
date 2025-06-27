from fastapi import APIRouter
from .dashboard import router as dashboard_router
# Create main router
router = APIRouter(prefix="/admin", tags=["Classes"])

# Include sub-routers without prefix to maintain the same URLs
router.include_router(dashboard_router)
