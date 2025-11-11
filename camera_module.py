import cv2
import numpy as np

class Camera:
    """Enhanced camera module with preprocessing and error handling."""
    
    def __init__(self, cam_index=0, width=640, height=480):
        """
        Initialize camera with specified resolution.
        
        Args:
            cam_index: Camera device index (default 0)
            width: Frame width (default 640)
            height: Frame height (default 480)
        """
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Webcam not detected. Please check your camera connection.")
        
        # Set camera properties for consistent performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.width = width
        self.height = height
        
        # Frame preprocessing settings
        self.denoise_strength = 7  # for fastNlMeansDenoisingColored
        
    def get_frame(self):
        """
        Capture and preprocess a frame.
        
        Returns:
            Preprocessed BGR frame or None on error
        """
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️  Failed to read frame from camera")
                return None
            
            if frame is None or frame.size == 0:
                print("⚠️  Received empty frame")
                return None
            
            # Ensure frame is properly formatted
            if len(frame.shape) != 3:
                print(f"⚠️  Frame has unexpected shape: {frame.shape}")
                return None
            
            # Optional: denoise frame for better edge detection
            # Commented out by default for performance, uncomment if needed
            # frame = cv2.fastNlMeansDenoisingColored(frame, None, self.denoise_strength, 
            #                                          self.denoise_strength, 7, 21)
            
            return frame
        
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return None
    
    def release(self):
        """Release camera resources and close windows."""
        self.cap.release()
        cv2.destroyAllWindows()
