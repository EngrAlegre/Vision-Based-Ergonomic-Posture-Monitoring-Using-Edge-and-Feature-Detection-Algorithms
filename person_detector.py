import cv2
import numpy as np

class PersonDetector:
    """
    Enhanced person detector using Haar cascades with improved efficiency.
    Detects face and extracts ROI for further processing.
    """
    
    def __init__(self):
        """Initialize Haar cascade classifiers."""
        # Load face cascade (most reliable for posture monitoring)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Upper body cascade for torso detection
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )
        
        # Detection parameters (tuned for desk sitting posture)
        self.scale_factor = 1.1
        self.min_neighbors_face = 4
        self.min_neighbors_body = 3
        
        # Store last known positions for stability
        self.last_face = None
        self.last_body = None
        self.smoothing_factor = 0.7  # 0 = no smoothing, 1 = full smoothing
        
    def _smooth_bbox(self, new_bbox, last_bbox, factor):
        """
        Apply temporal smoothing to bounding box coordinates.
        
        Args:
            new_bbox: New detected bounding box (x, y, w, h)
            last_bbox: Previous bounding box
            factor: Smoothing factor (0-1)
            
        Returns:
            Smoothed bounding box
        """
        if last_bbox is None:
            return new_bbox
        
        x, y, w, h = new_bbox
        lx, ly, lw, lh = last_bbox
        
        # Exponential moving average
        sx = int(factor * lx + (1 - factor) * x)
        sy = int(factor * ly + (1 - factor) * y)
        sw = int(factor * lw + (1 - factor) * w)
        sh = int(factor * lh + (1 - factor) * h)
        
        return (sx, sy, sw, sh)
    
    def detect(self, frame):
        """
        Detect person (face and body) in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            tuple: (annotated_frame, found, face_roi, body_roi)
                - annotated_frame: Frame with detection rectangles
                - found: Boolean indicating if person detected
                - face_roi: Face bounding box (x, y, w, h) or None
                - body_roi: Body bounding box (x, y, w, h) or None
        """
        # Convert to grayscale for Haar cascade (single conversion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (most important for posture monitoring)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors_face,
            minSize=(60, 60)
        )
        
        # Detect upper body
        bodies = self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors_body,
            minSize=(100, 100)
        )
        
        face_roi = None
        body_roi = None
        
        # Process face detection (prefer largest face)
        if len(faces) > 0:
            # Get largest face (closest to camera)
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = tuple(face)
            
            # Apply temporal smoothing
            face_roi = self._smooth_bbox(face_roi, self.last_face, self.smoothing_factor)
            self.last_face = face_roi
            
            # Draw face rectangle with professional styling
            x, y, w, h = face_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 0), 2)
            cv2.putText(frame, "HEAD", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
        
        # Process body detection (prefer largest body)
        if len(bodies) > 0:
            body = max(bodies, key=lambda rect: rect[2] * rect[3])
            body_roi = tuple(body)
            
            # Apply temporal smoothing
            body_roi = self._smooth_bbox(body_roi, self.last_body, self.smoothing_factor)
            self.last_body = body_roi
            
            # Note: Body ROI is stored but NOT drawn (hidden torso bounding box)
        
        # Person is detected if we have at least a face
        found = face_roi is not None
        
        return frame, found, face_roi, body_roi
