import cv2
import numpy as np
from collections import deque

class BasicPostureDetector:
    """
    Posture detector using ONLY basic computer vision techniques.
    No machine learning - uses contours, geometry, and color-based detection.
    
    Key Points Estimated:
    - Head center (from face detection)
    - Neck position (geometric estimation)
    - Shoulders (left/right from contour analysis)
    """
    
    def __init__(self):
        """Initialize detector with processing parameters."""
        # Skin detection parameters (YCrCb color space)
        self.skin_lower = np.array([0, 133, 77], dtype=np.uint8)
        self.skin_upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Morphological kernel sizes
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Temporal smoothing buffers (moving average)
        self.buffer_size = 5
        self.head_buffer = deque(maxlen=self.buffer_size)
        self.shoulder_left_buffer = deque(maxlen=self.buffer_size)
        self.shoulder_right_buffer = deque(maxlen=self.buffer_size)
        self.neck_buffer = deque(maxlen=self.buffer_size)
        self.torso_buffer = deque(maxlen=self.buffer_size)
        
        # ===== DYNAMIC LINE ANALYSIS =====
        # Head-neck line behavior tracking
        self.calibrated_neck_length = None  # Neutral reference length
        self.neck_line_buffer = deque(maxlen=10)  # Buffer for line length smoothing
        self.neck_angle_buffer = deque(maxlen=10)  # Buffer for angle smoothing
        self.calibration_frames = 30  # Frames to collect for calibration
        self.calibration_samples = []
        self.is_calibrated = False
        
        # Thresholds for dynamic line analysis
        self.LINE_SHORTEN_THRESHOLD = 0.60  # 60% of neutral length
        self.LINE_STRETCH_THRESHOLD = 1.25  # 125% of neutral length
        self.ANGLE_DEVIATION_THRESHOLD = 20  # Degrees from vertical
        
    def _smooth_point(self, point, buffer):
        """
        Apply temporal smoothing to a point using moving average.
        
        Args:
            point: New point (x, y)
            buffer: deque buffer for smoothing
            
        Returns:
            Smoothed point (x, y)
        """
        if point is None:
            return None
        
        buffer.append(point)
        
        if len(buffer) == 0:
            return point
        
        # Calculate average
        avg_x = int(np.mean([p[0] for p in buffer]))
        avg_y = int(np.mean([p[1] for p in buffer]))
        
        return (avg_x, avg_y)
    
    def _detect_skin_regions(self, frame):
        """
        Detect skin-colored regions using YCrCb color space.
        Useful for detecting exposed skin (face, neck, hands).
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of skin regions
        """
        # Convert to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Create binary mask for skin color range
        skin_mask = cv2.inRange(ycrcb, self.skin_lower, self.skin_upper)
        
        # Morphological operations to clean up noise
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        
        # Apply Gaussian blur to smooth edges
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        return skin_mask
    
    def _find_body_contour(self, frame, body_roi=None):
        """
        Find the main body contour using edge detection and filtering.
        
        Args:
            frame: Input BGR frame
            body_roi: Optional body bounding box to focus search
            
        Returns:
            Main body contour or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on ROI if provided
        if body_roi is not None:
            x, y, w, h = body_roi
            gray_roi = gray[y:y+h, x:x+w]
            offset = (x, y)
        else:
            gray_roi = gray
            offset = (0, 0)
        
        # Apply bilateral filter (preserves edges while reducing noise)
        filtered = cv2.bilateralFilter(gray_roi, 9, 75, 75)
        
        # Edge detection with Canny
        edges = cv2.Canny(filtered, 50, 150)
        
        # Morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Filter contours by area and aspect ratio
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Too small
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / w if w > 0 else 0
            
            # Human torso typically has aspect ratio between 1.0 and 3.0
            if 0.8 <= aspect_ratio <= 4.0:
                valid_contours.append(cnt)
        
        if len(valid_contours) == 0:
            return None
        
        # Select largest valid contour (most likely the body)
        body_contour = max(valid_contours, key=cv2.contourArea)
        
        # Adjust contour coordinates by offset
        if offset != (0, 0):
            body_contour = body_contour + np.array([offset], dtype=np.int32)
        
        return body_contour
    
    def calibrate_neck_length(self, head, neck):
        """
        Calibrate the neutral neck length from initial frames.
        Collects samples during good posture to establish baseline.
        
        Args:
            head: Head point (x, y)
            neck: Neck point (x, y)
        """
        if head is None or neck is None:
            return
        
        # Calculate current neck length
        length = np.sqrt((head[0] - neck[0])**2 + (head[1] - neck[1])**2)
        
        # Collect calibration samples
        if not self.is_calibrated:
            self.calibration_samples.append(length)
            
            if len(self.calibration_samples) >= self.calibration_frames:
                # Use median to avoid outliers
                self.calibrated_neck_length = np.median(self.calibration_samples)
                self.is_calibrated = True
                print(f"✓ Neck length calibrated: {self.calibrated_neck_length:.1f} pixels")
    
    def analyze_neck_line_dynamics(self, head, neck):
        """
        Analyze the dynamic behavior of the head-neck line.
        Detects shortening, bending, and angle deviations.
        
        Args:
            head: Head point (x, y)
            neck: Neck point (x, y)
            
        Returns:
            Dictionary with line analysis
        """
        result = {
            'length': None,
            'smoothed_length': None,
            'length_ratio': None,
            'is_shortened': False,
            'is_stretched': False,
            'angle_from_vertical': None,
            'smoothed_angle': None,
            'is_bent': False,
            'line_quality': 'unknown',
            'issues': []
        }
        
        if head is None or neck is None:
            return result
        
        # Calculate current line length (Euclidean distance)
        dx = head[0] - neck[0]
        dy = head[1] - neck[1]
        current_length = np.sqrt(dx**2 + dy**2)
        result['length'] = current_length
        
        # Apply temporal smoothing to length
        self.neck_line_buffer.append(current_length)
        smoothed_length = np.mean(list(self.neck_line_buffer))
        result['smoothed_length'] = smoothed_length
        
        # Calibrate if needed
        if not self.is_calibrated:
            self.calibrate_neck_length(head, neck)
            result['line_quality'] = 'calibrating'
            return result
        
        # Calculate length ratio relative to calibrated baseline
        length_ratio = smoothed_length / self.calibrated_neck_length
        result['length_ratio'] = length_ratio
        
        # Check for shortening (slouching/forward head)
        if length_ratio < self.LINE_SHORTEN_THRESHOLD:
            result['is_shortened'] = True
            result['issues'].append(f"Line shortened ({length_ratio*100:.0f}% of neutral)")
        
        # Check for excessive stretching (unnatural extension)
        if length_ratio > self.LINE_STRETCH_THRESHOLD:
            result['is_stretched'] = True
            result['issues'].append(f"Line stretched ({length_ratio*100:.0f}% of neutral)")
        
        # Calculate angle from vertical (0° = perfectly vertical)
        if abs(dy) > 1:  # Avoid division by zero
            angle_rad = np.arctan2(abs(dx), abs(dy))
            angle_deg = np.degrees(angle_rad)
            result['angle_from_vertical'] = angle_deg
            
            # Apply temporal smoothing to angle
            self.neck_angle_buffer.append(angle_deg)
            smoothed_angle = np.mean(list(self.neck_angle_buffer))
            result['smoothed_angle'] = smoothed_angle
            
            # Check for bending (deviation from vertical)
            if smoothed_angle > self.ANGLE_DEVIATION_THRESHOLD:
                result['is_bent'] = True
                result['issues'].append(f"Line bent ({smoothed_angle:.1f}° from vertical)")
        
        # Determine overall line quality
        if result['is_shortened'] or result['is_bent'] or result['is_stretched']:
            result['line_quality'] = 'poor'
        elif length_ratio < 0.85 or (result['smoothed_angle'] and result['smoothed_angle'] > 10):
            result['line_quality'] = 'warning'
        else:
            result['line_quality'] = 'good'
        
        return result
    
    def _detect_shoulders_from_body(self, frame, body_roi):
        """
        Detect shoulder positions independently using contour analysis on the body ROI.
        This avoids deriving shoulders from head position.
        
        Args:
            frame: Input BGR frame
            body_roi: Body bounding box (x, y, w, h)
            
        Returns:
            Tuple of (left_shoulder, right_shoulder) points or (None, None)
        """
        if body_roi is None:
            return None, None
        
        bx, by, bw, bh = body_roi
        
        # Extract body region
        body_region = frame[by:by+bh, bx:bx+bw]
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return None, None
            
            # Get the largest contour (body outline)
            body_contour = max(contours, key=cv2.contourArea)
            
            # Find extremes of the contour
            # Left extreme (leftmost point)
            leftmost = tuple(body_contour[body_contour[:, :, 0].argmin()][0])
            # Right extreme (rightmost point)
            rightmost = tuple(body_contour[body_contour[:, :, 0].argmax()][0])
            # Top extreme
            topmost = tuple(body_contour[body_contour[:, :, 1].argmin()][0])
            
            # Shoulders are typically near the top of the body contour
            # Use the extremes to estimate shoulder positions
            shoulder_y = topmost[1] + int(bh * 0.1)  # Slightly below top
            
            # Adjust coordinates to frame coordinates
            left_shoulder = (bx + leftmost[0], by + shoulder_y)
            right_shoulder = (bx + rightmost[0], by + shoulder_y)
            
            return left_shoulder, right_shoulder
            
        except Exception:
            return None, None
    
    def _estimate_neck_from_shoulders(self, left_shoulder, right_shoulder):
        """
        Estimate neck position as the midpoint between shoulders.
        This ensures neck is independent of head position.
        
        Args:
            left_shoulder: Left shoulder point (x, y)
            right_shoulder: Right shoulder point (x, y)
            
        Returns:
            Neck point (x, y) or None
        """
        if left_shoulder is None or right_shoulder is None:
            return None
        
        neck_x = (left_shoulder[0] + right_shoulder[0]) // 2
        neck_y = (left_shoulder[1] + right_shoulder[1]) // 2
        
        return (neck_x, neck_y)
    
    def estimate_keypoints(self, frame, face_roi, body_roi):
        """
        Estimate key body points with INDEPENDENT ANCHORING.
        
        Keypoints are not derived from each other:
        - Head: From face bounding box (bottom center)
        - Shoulders: From contour analysis of body region
        - Neck: Midpoint between shoulders (NOT from head)
        - Torso: Midpoint of shoulder line (NOT from head)
        
        This allows realistic head-neck line movement.
        
        Args:
            frame: Input BGR frame
            face_roi: Face bounding box (x, y, w, h)
            body_roi: Body bounding box (x, y, w, h) or None
        
        Returns:
            Dictionary of estimated keypoints with independent anchoring
        """
        keypoints = {}
        
        if face_roi is None:
            return keypoints
        
        fx, fy, fw, fh = face_roi
        
        # ===== 1. HEAD CENTER (Independent - from face bounding box) =====
        head_x = fx + fw // 2
        head_y = fy + fh  # Bottom of face (chin area)
        head = (head_x, head_y)
        keypoints['head'] = self._smooth_point(head, self.head_buffer)
        
        # ===== 2. SHOULDERS (Independent - from body contour analysis) =====
        left_shoulder, right_shoulder = self._detect_shoulders_from_body(frame, body_roi)
        
        # Fallback if contour detection fails
        if left_shoulder is None or right_shoulder is None:
            if body_roi is not None:
                bx, by, bw, bh = body_roi
                shoulder_y = by + int(bh * 0.1)
                shoulder_width = bw * 0.7
                left_shoulder = (bx + int(bw * 0.15), shoulder_y)
                right_shoulder = (bx + int(bw * 0.85), shoulder_y)
            else:
                # Fallback without body ROI
                shoulder_width = int(fw * 1.5)
                shoulder_y = head_y + int(fh * 0.5)
                left_shoulder = (head_x - shoulder_width // 2, shoulder_y)
                right_shoulder = (head_x + shoulder_width // 2, shoulder_y)
        
        keypoints['left_shoulder'] = self._smooth_point(left_shoulder, self.shoulder_left_buffer)
        keypoints['right_shoulder'] = self._smooth_point(right_shoulder, self.shoulder_right_buffer)
        
        # ===== 3. NECK (Independent - midpoint of shoulders, NOT from head) =====
        neck = self._estimate_neck_from_shoulders(
            keypoints['left_shoulder'], 
            keypoints['right_shoulder']
        )
        
        if neck is None:
            # Fallback: use average of raw shoulder positions
            neck = self._estimate_neck_from_shoulders(left_shoulder, right_shoulder)
        
        if neck is not None:
            keypoints['neck'] = self._smooth_point(neck, self.neck_buffer)
        
        # ===== 4. TORSO CENTER (Independent - midpoint of shoulders, below neck) =====
        if keypoints.get('left_shoulder') and keypoints.get('right_shoulder'):
            torso_x = (keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) // 2
            torso_y = keypoints['left_shoulder'][1] + int(fh * 1.5)
            torso = (torso_x, torso_y)
            keypoints['torso_center'] = self._smooth_point(torso, self.torso_buffer)
        elif body_roi is not None:
            bx, by, bw, bh = body_roi
            torso = (bx + bw // 2, by + bh // 2)
            keypoints['torso_center'] = self._smooth_point(torso, self.torso_buffer)
        
        return keypoints
    
    def draw_skeleton(self, frame, keypoints):
        """
        Draw skeletal overlay with dynamic head-neck line analysis.
        
        Args:
            frame: Input frame to draw on
            keypoints: Dictionary of keypoints from estimate_keypoints()
            
        Returns:
            Frame with skeleton overlay
        """
        if not keypoints:
            return frame
        
        head = keypoints.get('head')
        neck = keypoints.get('neck')
        
        # ===== DYNAMIC HEAD-NECK LINE ANALYSIS =====
        if head is not None and neck is not None:
            # Analyze line dynamics
            line_analysis = self.analyze_neck_line_dynamics(head, neck)
            
            # Determine line color based on quality
            quality = line_analysis.get('line_quality', 'unknown')
            
            if quality == 'good':
                line_color = (0, 255, 0)  # Green
                line_thickness = 3
            elif quality == 'warning':
                line_color = (0, 165, 255)  # Orange
                line_thickness = 4
            elif quality == 'poor':
                line_color = (0, 0, 255)  # Red
                line_thickness = 5
            elif quality == 'calibrating':
                line_color = (255, 255, 0)  # Yellow
                line_thickness = 3
            else:
                line_color = (128, 128, 128)  # Gray
                line_thickness = 2
            
            # Draw dynamic head-neck line
            cv2.line(frame, head, neck, line_color, line_thickness, cv2.LINE_AA)
            
            # Draw endpoint circles
            cv2.circle(frame, head, 8, line_color, -1)
            cv2.circle(frame, head, 10, (255, 255, 255), 2)
            cv2.circle(frame, neck, 8, line_color, -1)
            cv2.circle(frame, neck, 10, (255, 255, 255), 2)
            
            # Store analysis for external use
            self.last_line_analysis = line_analysis
            
            # Draw metrics
            if self.is_calibrated and line_analysis.get('length_ratio') is not None:
                length_ratio = line_analysis['length_ratio']
                angle = line_analysis.get('smoothed_angle')
                
                # Only draw if angle is valid
                if angle is not None:
                    metrics_text = f"Length: {length_ratio*100:.0f}% | Angle: {angle:.1f}°"
                else:
                    metrics_text = f"Length: {length_ratio*100:.0f}%"
                
                cv2.putText(frame, metrics_text, (neck[0] + 10, neck[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)
            elif quality == 'calibrating':
                remaining = self.calibration_frames - len(self.calibration_samples)
                cal_text = f"Calibrating... {remaining} frames"
                cv2.putText(frame, cal_text, (head[0] + 10, head[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)
        
        # Define skeleton connections (excluding head-neck)
        connections = [
            ('neck', 'left_shoulder'),
            ('neck', 'right_shoulder'),
            ('left_shoulder', 'torso_center'),
            ('right_shoulder', 'torso_center')
        ]
        
        # Draw connections (bones)
        for start_key, end_key in connections:
            if start_key in keypoints and end_key in keypoints:
                start = keypoints[start_key]
                end = keypoints[end_key]
                if start is not None and end is not None:
                    # Draw bone line
                    cv2.line(frame, start, end, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw keypoints (joints) - except head/neck which are handled above
        for key, point in keypoints.items():
            if point is not None and key not in ['head', 'neck']:
                x, y = point
                # Draw joint circle
                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
                cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)
                
                # Label keypoint
                label = key.replace('_', ' ').title()
                cv2.putText(frame, label, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def analyze_posture(self, keypoints):
        """
        Analyze posture based on DYNAMIC LINE ANALYSIS.
        
        Primary metric: Head-neck line behavior
        - Length: Indicates slouching vs extension
        - Angle: Indicates neck tilt vs straight alignment
        
        Secondary metrics: Shoulder alignment
        
        Args:
            keypoints: Dictionary of keypoints with independent anchoring
        
        Returns:
            Dictionary with analysis results including line quality
        """
        result = {
            'score': 100,
            'issues': [],
            'is_poor': False,
            'line_quality': 'unknown'
        }
        
        head = keypoints.get('head')
        neck = keypoints.get('neck')
        
        # ===== PRIMARY: DYNAMIC HEAD-NECK LINE ANALYSIS =====
        if hasattr(self, 'last_line_analysis') and head is not None and neck is not None:
            line_analysis = self.last_line_analysis
            result['line_quality'] = line_analysis.get('line_quality', 'unknown')
            
            # During calibration, be lenient
            if result['line_quality'] == 'calibrating':
                result['score'] = 100
                result['is_poor'] = False
                return result
            
            # Apply penalties based on dynamic line behavior
            if line_analysis.get('is_shortened'):
                result['issues'].append("Slouching detected (line shortened)")
                result['score'] -= 40
            
            if line_analysis.get('is_bent'):
                result['issues'].append("Neck tilt detected (line bent)")
                result['score'] -= 40
            
            if line_analysis.get('is_stretched'):
                result['issues'].append("Over-extension (line stretched)")
                result['score'] -= 15
            
            # Add ratio-based penalty for warning threshold
            length_ratio = line_analysis.get('length_ratio')
            if length_ratio and 0.85 > length_ratio >= 0.60:
                result['issues'].append("Posture degrading")
                result['score'] -= 20
            
            # Add angle-based penalty for warning threshold
            angle = line_analysis.get('smoothed_angle')
            if angle and 20 > angle > 10:
                result['issues'].append("Slight neck misalignment")
                result['score'] -= 15
        
        # ===== SECONDARY: SHOULDER ALIGNMENT =====
        left_shoulder = keypoints.get('left_shoulder')
        right_shoulder = keypoints.get('right_shoulder')
        
        if left_shoulder and right_shoulder:
            shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
            if shoulder_tilt > 25:  # Higher threshold for robustness
                result['issues'].append("Uneven shoulders")
                result['score'] -= 20
        
        # ===== FINALIZE SCORE =====
        result['score'] = max(0, min(100, result['score']))
        result['is_poor'] = result['score'] < 70
        
        return result
