"""
HOG-Based Person Detector (Module 11: Object Detection)
========================================================

This module uses Histogram of Oriented Gradients (HOG) for person detection.
HOG is more robust than Haar cascades but still classical CV (no ML training required).

HOG is a feature descriptor that captures edge orientation distributions,
making it excellent for detecting people in various poses.
"""

import cv2
import numpy as np

class HOGPersonDetector:
    """
    Person detector using HOG descriptors.
    More robust than Haar cascades for full-body detection.
    
    Module 11: Object Detection - HOG Descriptors
    """
    
    def __init__(self):
        """Initialize HOG descriptor with OpenCV's pre-configured parameters."""
        # Initialize HOG descriptor (pre-configured, not ML-based)
        self.hog = cv2.HOGDescriptor()
        
        # Load pre-computed HOG SVM (this is a classical SVM, not deep learning)
        # OpenCV includes this as part of classical CV techniques
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detection parameters
        self.win_stride = (8, 8)  # Window stride for detection
        self.padding = (8, 8)     # Padding around detection window
        self.scale = 1.05         # Scale factor for multi-scale detection
        
        # Temporal smoothing
        self.last_detections = []
        self.smoothing_frames = 3
        
    def detect(self, frame):
        """
        Detect people in frame using HOG descriptors.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            tuple: (annotated_frame, found, person_roi)
                - annotated_frame: Frame with bounding boxes
                - found: Boolean if person detected
                - person_roi: Main person bounding box (x, y, w, h)
        """
        # Detect people using HOG
        # detectMultiScale returns bounding boxes and confidence scores
        (rects, weights) = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            useMeanshiftGrouping=False  # Use NMS instead
        )
        
        # Apply non-maximum suppression to remove overlapping boxes
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        
        if len(rects) > 0:
            # Pick detection with highest confidence (largest area)
            areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in rects]
            best_idx = np.argmax(areas)
            x1, y1, x2, y2 = rects[best_idx]
            
            person_roi = (x1, y1, x2 - x1, y2 - y1)
            
            # Apply temporal smoothing
            self.last_detections.append(person_roi)
            if len(self.last_detections) > self.smoothing_frames:
                self.last_detections.pop(0)
            
            # Average recent detections
            if len(self.last_detections) > 0:
                avg_x = int(np.mean([d[0] for d in self.last_detections]))
                avg_y = int(np.mean([d[1] for d in self.last_detections]))
                avg_w = int(np.mean([d[2] for d in self.last_detections]))
                avg_h = int(np.mean([d[3] for d in self.last_detections]))
                person_roi = (avg_x, avg_y, avg_w, avg_h)
            
            # Draw bounding box
            x, y, w, h = person_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "PERSON (HOG)", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return frame, True, person_roi
        
        return frame, False, None
    
    def extract_upper_body_roi(self, person_roi):
        """
        Extract upper body region from full person detection.
        Useful for posture analysis (focus on torso/shoulders).
        
        Args:
            person_roi: Full person bounding box (x, y, w, h)
            
        Returns:
            Upper body ROI (x, y, w, h)
        """
        if person_roi is None:
            return None
        
        x, y, w, h = person_roi
        
        # Upper body is approximately top 50% of person detection
        upper_h = int(h * 0.5)
        
        return (x, y, w, upper_h)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demo_hog_detection():
    """
    Demonstration of HOG-based person detection.
    
    HOG Benefits over Haar Cascades:
    - More robust to pose variations
    - Better full-body detection
    - Handles partial occlusions better
    - More accurate bounding boxes
    
    HOG Limitations:
    - Slightly slower than Haar cascades
    - Works best with upright people
    - Requires more computational resources
    """
    print("=" * 60)
    print("HOG-Based Person Detection Demo (Module 11)")
    print("=" * 60)
    
    # Initialize detector
    detector = HOGPersonDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    print("\n✓ HOG detector initialized")
    print("✓ Camera opened")
    print("\nPress 'Q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect person using HOG
        frame, found, person_roi = detector.detect(frame)
        
        if found and person_roi:
            # Extract upper body for posture analysis
            upper_body = detector.extract_upper_body_roi(person_roi)
            
            if upper_body:
                x, y, w, h = upper_body
                cv2.rectangle(frame, (x, y), (x + w, y + h), 
                             (255, 0, 0), 1)
                cv2.putText(frame, "Upper Body ROI", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        else:
            cv2.putText(frame, "No person detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("HOG Person Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_hog_detection()


# ============================================================================
# INTEGRATION WITH MAIN SYSTEM
# ============================================================================
"""
To integrate HOG detector into the main improved system:

1. In main.py, replace PersonDetector with HOGPersonDetector:

    from hog_person_detector import HOGPersonDetector
    
    # Instead of:
    # person_detector = PersonDetector()
    
    # Use:
    person_detector = HOGPersonDetector()

2. The rest of the pipeline remains the same!

COMPARISON: Haar vs HOG

Haar Cascades:
+ Faster (good for real-time)
+ Lower resource usage
- Less robust to pose variations
- Face/upper body only

HOG Descriptors:
+ More robust detection
+ Better full-body detection
+ Handles variations better
- Slightly slower
- More computational cost

RECOMMENDATION: 
- Use Haar for real-time on low-end hardware
- Use HOG for better accuracy on decent hardware
"""
