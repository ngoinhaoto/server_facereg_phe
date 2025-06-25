import time
import os
import random
import json
import io
from locust import HttpUser, task, between, TaskSet
import base64

# Configure sample face image paths
SAMPLE_IMAGES_DIR = "tests/performance/sample_faces"

# Ensure the sample image directory exists
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

# Admin credentials for authentication
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def load_sample_image(filename):
    """Load a sample image as binary data (not base64)"""
    try:
        with open(os.path.join(SAMPLE_IMAGES_DIR, filename), "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading sample image {filename}: {e}")
        return None


class CheckInPerformanceTest(HttpUser):    
    # Wait between 0.5 and 2 seconds between tasks - shorter wait for more load
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Authenticate and load sample images when user starts"""
        # Authenticate with admin credentials
        self.token = self.authenticate()
        if not self.token:
            print("WARNING: Authentication failed! Tests will likely fail.")
        else:
            print(f"Successfully authenticated as admin with token: {self.token[:10]}...")
        
        # Load sample images
        self.sample_images = []
        
        image_files = [f for f in os.listdir(SAMPLE_IMAGES_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"WARNING: No sample images found in {SAMPLE_IMAGES_DIR}")
            print("Please add sample face images to this directory for testing")
        else:
            print(f"Loaded {len(image_files)} sample images for testing")
            for filename in image_files:
                img_data = load_sample_image(filename)
                if img_data:
                    self.sample_images.append((filename, img_data))
    
    def authenticate(self):
        """Authenticate with the API and return the token"""
        try:
            login_data = {
                "username": ADMIN_USERNAME,
                "password": ADMIN_PASSWORD
            }
            
            with self.client.post(
                "/auth/token", 
                data=login_data,  # Form data, not JSON
                catch_response=True,
                name="Login"
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    if "access_token" in result:
                        return result["access_token"]
                    else:
                        print(f"Authentication response missing token: {result}")
                else:
                    print(f"Authentication failed with status {response.status_code}: {response.text}")
                
                return None
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return None
    
    def get_random_image(self):
        """Get a random sample image with its filename"""
        if not self.sample_images:
            raise Exception("No sample images available. Add images to the sample directory.")
        return random.choice(self.sample_images)
    
    @task
    def test_check_in(self):
        """Test the check-in endpoint performance"""
        if not self.sample_images or not hasattr(self, 'token') or not self.token:
            print("Skipping check-in test: missing images or authentication token")
            return
            
        # Get a random image and its filename for better tracking
        filename, image_data = self.get_random_image()
        
        # Set up authorization header with the token
        headers = {
            "Authorization": f"Bearer {self.token}"
            # Don't set Content-Type - requests will set it automatically for multipart form
        }
        
        # Set up query parameters - session_id needs to be in the URL query
        params = {
            "session_id": "1",  # Using string as query params are typically strings
            "device_id": "locust_perf_test"
        }
        
        # Prepare the file upload
        files = {
            'file': (filename, image_data, 'image/jpeg')  # (filename, file_data, content_type)
        }
        
        # Print first request for debugging
        if not hasattr(self, 'first_request_printed'):
            print(f"Sample request: POST /attendance/check-in?session_id={params['session_id']}&device_id={params['device_id']} with file upload")
            self.first_request_printed = True
        
        # Record the start time to calculate response time
        start_time = time.time()
        
        # Make the request - using files parameter for multipart/form-data
        with self.client.post(
            "/attendance/check-in", 
            files=files,
            params=params,  # Add query parameters
            headers=headers,
            catch_response=True,
            name=f"Check-in (Image: {filename})"  # This helps track per-image performance
        ) as response:
            # Calculate response time
            response_time = time.time() - start_time
            
            try:
                # Print full response for the first error to debug
                if response.status_code != 200 and not hasattr(self, 'first_error_printed'):
                    print(f"First error response: {response.text}")
                    self.first_error_printed = True
                
                result = response.json()
                
                if response.status_code == 200:
                    # Success - add custom metrics if available in the response
                    if "processing_time" in result:
                        print(f"Check-in processing time: {result['processing_time']}ms")
                    
                    # Log which service was used if available
                    service_info = ""
                    if "service" in result:
                        service_info = f" using {result['service']}"
                    
                    # Check if user was recognized
                    if "user_recognized" in result and result["user_recognized"]:
                        # Just call success() without parameters
                        response.success()
                        print(f"User recognized{service_info} in {response_time:.2f}s")
                    else:
                        # Just call success() without parameters
                        response.success()
                        print(f"Check-in successful{service_info} in {response_time:.2f}s")
                elif response.status_code == 422:
                    # Validation error - print detailed information to help fix the request
                    validation_errors = result.get("detail", [])
                    error_details = []
                    for error in validation_errors:
                        field = ".".join(str(loc) for loc in error.get("loc", []))
                        msg = error.get("msg", "Unknown validation error")
                        error_details.append(f"{field}: {msg}")
                    
                    error_msg = ", ".join(error_details)
                    # Just call failure() without parameters or with a string
                    response.failure(f"Validation error: {error_msg}")
                    print(f"Validation error details: {json.dumps(validation_errors, indent=2)}")
                elif response.status_code == 401 or response.status_code == 403:
                    # Authentication error - try to refresh token
                    error_msg = f"Authentication error ({response.status_code}): {result.get('detail', 'Unknown auth error')}"
                    response.failure(error_msg)
                    print("Attempting to refresh authentication token...")
                    self.token = self.authenticate()
                else:
                    # Handle error cases
                    error_msg = result.get("error", result.get("detail", "Unknown error"))
                    response.failure(f"Check-in failed: {error_msg} ({response_time:.2f}s)")
                    
            except Exception as e:
                response.failure(f"Invalid response: {str(e)} ({response_time:.2f}s)")