"""
Hough Line Transform for Spine Alignment Detection (Module 4)
==============================================================

This module uses Hough Line Transform to detect the spine/body centerline
and assess posture alignment.

Hough Transform is a classical CV technique for detecting geometric shapes
(lines, circles) in images.
"""

import cv2
import numpy as np
from collections import deque

class SpineAlignmentDetector:
    """
    Detect spine alignment using Hough Line Transform (Module 4).
    Analyzes body centerline for posture assessment.
    """
    
    def __init__(self):
        """Initialize spine alignment detector."""
        # Hough Line parameters
        self.rho = 1  # Distance resolution
        self.theta = np.pi / 180  # Angle resolution (1 degree)
        self.threshold = 50  # Minimum votes
        self.min_line_length = 100  # Minimum line length (pixels)
        self.max_line_gap = 20  # Maximum gap between line segments
        
        # Temporal smoothing
        self.spine_angle_buffer = deque(maxlen=5)
        
        # Ideal spine angle (degrees from vertical)
        self.IDEAL_SPINE_ANGLE = 0  # Perfectly vertical
        self.GOOD_SPINE_RANGE = 15  # ±15 degrees is acceptable
        
    def extract_body_centerline(self, edges, body_roi):
        """
        Extract body centerline region for spine detection.
        
        Args:
            edges: Edge-detected image
            body_roi: Body bounding box (x, y, w, h)
            
        Returns:
            Centerline region mask
        """
        if body_roi is None:
            return edges
        
        x, y, w, h = body_roi
        
        # Create mask for body center (middle 40% horizontally)
        mask = np.zeros_like(edges)
        center_x = x + w // 2
        center_w = int(w * 0.4)  # 40% of body width
        
        mask[y:y+h, center_x - center_w//2:center_x + center_w//2] = 255
        
        # Apply mask to edges
        centerline = cv2.bitwise_and(edges, mask)
        
        return centerline
    
    def detect_spine_line(self, edges, body_roi):
        """
        Detect the main spine/body centerline using Hough Line Transform.
        
        Args:
            edges: Edge-detected image
            body_roi: Body bounding box (x, y, w, h)
            
        Returns:
            tuple: (spine_line, spine_angle)
                spine_line: Line endpoints [(x1, y1), (x2, y2)] or None
                spine_angle: Angle from vertical in degrees
        """
        if body_roi is None:
            return None, None
        
        # Extract body centerline region
        centerline = self.extract_body_centerline(edges, body_roi)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            centerline,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None or len(lines) == 0:
            return None, None
        
        # Filter lines: prefer vertical lines near body center
        x, y, w, h = body_roi
        body_center_x = x + w // 2
        
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Skip very short lines
            if length < self.min_line_length:
                continue
            
            # Calculate angle from vertical
            dx = x2 - x1
            dy = y2 - y1
            
            if dy == 0:  # Horizontal line, skip
                continue
            
            # Angle from vertical (0 degrees = perfectly vertical)
            angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
            
            # Distance from body center
            line_center_x = (x1 + x2) / 2
            dist_from_center = abs(line_center_x - body_center_x)
            
            # Score: prefer vertical lines near center
            score = length / (1 + angle) / (1 + dist_from_center)
            
            vertical_lines.append({
                'line': [(x1, y1), (x2, y2)],
                'angle': angle if dx >= 0 else -angle,  # Sign indicates direction
                'score': score,
                'length': length
            })
        
        if len(vertical_lines) == 0:
            return None, None
        
        # Select best line (highest score)
        best_line = max(vertical_lines, key=lambda l: l['score'])
        
        # Apply temporal smoothing to angle
        self.spine_angle_buffer.append(best_line['angle'])
        smoothed_angle = np.mean(list(self.spine_angle_buffer))
        
        return best_line['line'], smoothed_angle
    
    def analyze_spine_alignment(self, spine_angle):
        """
        Analyze spine alignment quality.
        
        Args:
            spine_angle: Angle from vertical in degrees
            
        Returns:
            Dictionary with alignment analysis
        """
        if spine_angle is None:
            return {
                'is_aligned': False,
                'alignment_quality': 'Unknown',
                'deviation': None,
                'message': 'Cannot detect spine line'
            }
        
        deviation = abs(spine_angle - self.IDEAL_SPINE_ANGLE)
        
        # Determine alignment quality
        if deviation <= self.GOOD_SPINE_RANGE:
            quality = 'Good'
            is_aligned = True
            message = 'Spine is well aligned'
        elif deviation <= self.GOOD_SPINE_RANGE * 2:
            quality = 'Fair'
            is_aligned = False
            message = f'Slight lean ({spine_angle:.1f}°)'
        else:
            quality = 'Poor'
            is_aligned = False
            
            if spine_angle > 0:
                message = f'Leaning right ({spine_angle:.1f}°)'
            else:
                message = f'Leaning left ({abs(spine_angle):.1f}°)'
        
        return {
            'is_aligned': is_aligned,
            'alignment_quality': quality,
            'deviation': deviation,
            'angle': spine_angle,
            'message': message
        }
    
    def visualize_spine_alignment(self, frame, spine_line, alignment_result):
        """
        Draw spine line and alignment information on frame.
        
        Args:
            frame: Input frame
            spine_line: Detected spine line [(x1, y1), (x2, y2)]
            alignment_result: Result from analyze_spine_alignment
            
        Returns:
            Annotated frame
        """
        # Draw spine line if detected
        if spine_line is not None:
            (x1, y1), (x2, y2) = spine_line
            
            # Color based on alignment quality
            quality = alignment_result.get('alignment_quality', 'Unknown')
            if quality == 'Good':
                color = (0, 255, 0)  # Green
            elif quality == 'Fair':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw thick spine line
            cv2.line(frame, (x1, y1), (x2, y2), color, 4)
            
            # Draw reference vertical line (ideal)
            mid_x = (x1 + x2) // 2
            cv2.line(frame, (mid_x, y1), (mid_x, y2), (255, 255, 255), 1, 
                    lineType=cv2.LINE_AA)
            
            # Draw angle indicator
            angle = alignment_result.get('angle', 0)
            cv2.putText(frame, f"Spine: {angle:.1f}°", (mid_x + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demo_spine_alignment():
    """
    Demonstration of Hough Line-based spine alignment detection.
    
    How it works:
    1. Detect edges in the frame
    2. Extract body centerline region
    3. Use Hough Line Transform to find dominant vertical line
    4. Calculate angle from perfect vertical
    5. Assess alignment quality
    
    Benefits:
    - Objective spine alignment measurement
    - Detects lateral (side-to-side) leaning
    - Complements forward/backward posture metrics
    - Uses classical CV (Hough Transform)
    
    Limitations:
    - Requires visible body edges (good contrast)
    - Works best with upright sitting/standing
    - May be affected by clothing patterns
    """
    print("=" * 60)
    print("Hough Line Spine Alignment Demo (Module 4)")
    print("=" * 60)
    
    detector = SpineAlignmentDetector()
    cap = cv2.VideoCapture(0)
    
    # Simple person detector (replace with Haar/HOG in production)
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    print("\n✓ Spine alignment detector initialized")
    print("✓ Camera opened")
    print("\nPress 'Q' to quit")
    print("\nTip: Sit upright and ensure your torso is visible!\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get person ROI (simplified - replace with better detection)
        fg_mask = backSub.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        body_roi = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5000:
                body_roi = cv2.boundingRect(largest)
                x, y, w, h = body_roi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect spine line
        spine_line, spine_angle = detector.detect_spine_line(edges, body_roi)
        
        # Analyze alignment
        alignment = detector.analyze_spine_alignment(spine_angle)
        
        # Visualize
        frame = detector.visualize_spine_alignment(frame, spine_line, alignment)
        
        # Display alignment info
        y_offset = 30
        quality = alignment['alignment_quality']
        color = (0, 255, 0) if quality == 'Good' else (0, 165, 255) if quality == 'Fair' else (0, 0, 255)
        
        cv2.putText(frame, f"Alignment: {quality}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, alignment['message'], (10, y_offset + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show edges in corner
        edges_small = cv2.resize(edges, (frame.shape[1]//4, frame.shape[0]//4))
        edges_color = cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR)
        frame[0:edges_small.shape[0], 0:edges_small.shape[1]] = edges_color
        
        cv2.imshow("Spine Alignment Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_spine_alignment()


# ============================================================================
# INTEGRATION WITH MAIN SYSTEM
# ============================================================================
"""
INTEGRATION: Add Spine Alignment as 5th Posture Metric

In posture_analyzer.py, add spine alignment analysis:

    from spine_alignment_detector import SpineAlignmentDetector
    
    class EnhancedPostureAnalyzer:
        def __init__(self):
            # ... existing code ...
            self.spine_detector = SpineAlignmentDetector()
        
        def analyze_posture(self, keypoints, frame, edges, body_roi):
            # ... existing metrics ...
            
            # METRIC 5: Spine Alignment (lateral)
            spine_line, spine_angle = self.spine_detector.detect_spine_line(
                edges, body_roi
            )
            spine_alignment = self.spine_detector.analyze_spine_alignment(
                spine_angle
            )
            
            result['spine_alignment'] = spine_alignment
            
            if not spine_alignment['is_aligned']:
                result['issues'].append(spine_alignment['message'])
                result['score'] -= 15

BENEFITS:
- Adds lateral (side-to-side) posture assessment
- Complements forward/backward metrics
- Uses classical Hough Transform (Module 4)
- Objective measurement of spinal alignment

NEW METRICS TOTAL:
1. Neck angle (existing)
2. Forward head (existing)
3. Shoulder alignment (existing)
4. Vertical alignment (existing)
5. Spine alignment (NEW - Hough Line)
"""
