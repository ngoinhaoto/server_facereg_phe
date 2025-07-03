from services.face_recognition.base import FaceRecognitionBase
from services.face_recognition.deepface_service import DeepFaceService

# Re-export the FaceRecognitionBase class with factory method for backward compatibility
class FaceRecognitionService(FaceRecognitionBase):
    """
    Backward compatibility wrapper for the refactored face recognition service.
    This class simply re-exports the FaceRecognitionBase class to maintain 
    the same import name in existing code.
    """
    pass

get_face_recognition_service = FaceRecognitionBase.get_instance