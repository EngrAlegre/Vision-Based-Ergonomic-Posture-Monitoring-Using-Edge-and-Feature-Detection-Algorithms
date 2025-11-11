import numpy as np
from collections import deque

class EnhancedPostureAnalyzer:
    """
    Posture analyzer using multiple geometric metrics.
    Provides comprehensive posture assessment using only basic CV techniques.
    """
    
    def __init__(self):
        """Initialize analyzer with configurable thresholds."""
        # Angle thresholds (degrees)
        self.GOOD_NECK_ANGLE_MIN = 150  # Neck should be relatively straight
        self.FORWARD_HEAD_THRESHOLD = 30  # Max forward head displacement (pixels)
        self.SHOULDER_LEVEL_THRESHOLD = 15  # Max shoulder height difference (pixels)
        
        # Temporal smoothing for angles
        self.angle_buffer = deque(maxlen=10)
        
        # Session statistics
        self.total_measurements = 0
        self.poor_posture_count = 0
        
    def calculate_angle(self, point_a, point_b, point_c):
        """
        Calculate angle ABC using three points.
        
        Args:
            point_a: First point (x, y)
            point_b: Vertex point (x, y)
            point_c: Third point (x, y)
            
        Returns:
            Angle in degrees (0-180)
        """
        if None in [point_a, point_b, point_c]:
            return None
        
        # Convert to numpy arrays
        a = np.array(point_a, dtype=np.float32)
        b = np.array(point_b, dtype=np.float32)
        c = np.array(point_c, dtype=np.float32)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)  # Clamp to valid range
        angle = np.degrees(np.arccos(cosine))
        
        return angle
    
    def calculate_forward_head_posture(self, head, neck, torso):
        """
        Calculate forward head displacement.
        Measures how far forward the head is relative to the torso centerline.
        
        Args:
            head: Head position (x, y)
            neck: Neck position (x, y)
            torso: Torso center position (x, y)
            
        Returns:
            Forward displacement in pixels (positive = forward lean)
        """
        if None in [head, neck, torso]:
            return None
        
        # Calculate horizontal displacement
        # Positive value means head is forward of torso centerline
        head_x = head[0]
        torso_x = torso[0]
        
        displacement = torso_x - head_x
        
        return displacement
    
    def calculate_shoulder_alignment(self, left_shoulder, right_shoulder):
        """
        Check if shoulders are level (not tilted).
        
        Args:
            left_shoulder: Left shoulder position (x, y)
            right_shoulder: Right shoulder position (x, y)
            
        Returns:
            Absolute height difference in pixels
        """
        if None in [left_shoulder, right_shoulder]:
            return None
        
        height_diff = abs(left_shoulder[1] - right_shoulder[1])
        return height_diff
    
    def analyze_posture(self, keypoints):
        """
        Comprehensive posture analysis using multiple metrics.
        
        Args:
            keypoints: Dictionary of body keypoints
            
        Returns:
            Dictionary containing:
            {
                'is_poor_posture': bool,
                'neck_angle': float or None,
                'forward_head': float or None,
                'shoulder_tilt': float or None,
                'issues': list of strings describing problems,
                'score': int (0-100, higher is better)
            }
        """
        result = {
            'is_poor_posture': False,
            'neck_angle': None,
            'forward_head': None,
            'shoulder_tilt': None,
            'issues': [],
            'score': 100
        }
        
        # Extract keypoints
        head = keypoints.get('head')
        neck = keypoints.get('neck')
        left_shoulder = keypoints.get('left_shoulder')
        right_shoulder = keypoints.get('right_shoulder')
        torso = keypoints.get('torso_center')
        
        # Can't analyze without minimum keypoints
        if None in [head, neck, torso]:
            return result
        
        # --- METRIC 1: Neck Angle (Primary metric) ---
        # Calculate angle: head -> neck -> torso
        neck_angle = self.calculate_angle(head, neck, torso)
        
        if neck_angle is not None:
            # Apply temporal smoothing
            self.angle_buffer.append(neck_angle)
            smoothed_angle = np.mean(list(self.angle_buffer))
            result['neck_angle'] = smoothed_angle
            
            # Check if angle indicates poor posture
            if smoothed_angle < self.GOOD_NECK_ANGLE_MIN:
                result['issues'].append(f"Forward neck tilt ({smoothed_angle:.1f}°)")
                result['score'] -= 30
        
        # --- METRIC 2: Forward Head Posture ---
        forward_head = self.calculate_forward_head_posture(head, neck, torso)
        
        if forward_head is not None:
            result['forward_head'] = forward_head
            
            # Positive value means head is behind torso (good)
            # Negative value means head is forward (bad)
            if forward_head < -self.FORWARD_HEAD_THRESHOLD:
                result['issues'].append(f"Head too far forward ({abs(forward_head):.0f}px)")
                result['score'] -= 25
        
        # --- METRIC 3: Shoulder Alignment ---
        if left_shoulder and right_shoulder:
            shoulder_tilt = self.calculate_shoulder_alignment(left_shoulder, right_shoulder)
            
            if shoulder_tilt is not None:
                result['shoulder_tilt'] = shoulder_tilt
                
                if shoulder_tilt > self.SHOULDER_LEVEL_THRESHOLD:
                    result['issues'].append(f"Uneven shoulders ({shoulder_tilt:.0f}px)")
                    result['score'] -= 20
        
        # --- METRIC 4: Overall Vertical Alignment ---
        # Check if head, neck, and torso form a relatively vertical line
        if head and neck and torso:
            # Calculate horizontal deviation from vertical line
            head_x = head[0]
            torso_x = torso[0]
            horizontal_deviation = abs(head_x - torso_x)
            
            if horizontal_deviation > 50:  # More than 50 pixels off-center
                result['issues'].append(f"Body not centered")
                result['score'] -= 15
        
        # Clamp score to 0-100 range
        result['score'] = max(0, min(100, result['score']))
        
        # Determine if posture is poor (score below 70 or has critical issues)
        result['is_poor_posture'] = result['score'] < 70 or len(result['issues']) >= 2
        
        # Update statistics
        self.total_measurements += 1
        if result['is_poor_posture']:
            self.poor_posture_count += 1
        
        return result
    
    def get_statistics(self):
        """
        Get session statistics.
        
        Returns:
            Dictionary with good/poor posture percentages
        """
        if self.total_measurements == 0:
            return {'good_pct': 0, 'poor_pct': 0}
        
        poor_pct = (self.poor_posture_count / self.total_measurements) * 100
        good_pct = 100 - poor_pct
        
        return {
            'good_pct': good_pct,
            'poor_pct': poor_pct,
            'total': self.total_measurements
        }
    
    def reset_statistics(self):
        """Reset session statistics."""
        self.total_measurements = 0
        self.poor_posture_count = 0
