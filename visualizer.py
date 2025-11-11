import cv2
import numpy as np

class PostureVisualizer:
    """
    Professional visualization for posture monitoring system.
    Creates clean, informative overlays without clutter.
    """
    
    def __init__(self, frame_width=640, frame_height=480):
        """
        Initialize visualizer with frame dimensions.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Color scheme (BGR format)
        self.COLOR_GOOD = (0, 255, 0)  # Green
        self.COLOR_POOR = (0, 0, 255)  # Red
        self.COLOR_WARNING = (0, 165, 255)  # Orange
        self.COLOR_INFO = (255, 255, 255)  # White
        self.COLOR_BACKGROUND = (0, 0, 0)  # Black
        self.COLOR_PANEL = (40, 40, 40)  # Dark gray
        
        # Font settings
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
        
    def _draw_rounded_rectangle(self, frame, top_left, bottom_right, color, thickness=2, radius=10):
        """
        Draw a rectangle with rounded corners.
        
        Args:
            frame: Frame to draw on
            top_left: Top-left corner (x, y)
            bottom_right: Bottom-right corner (x, y)
            color: BGR color
            thickness: Line thickness (-1 for filled)
            radius: Corner radius
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Draw straight lines
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corner arcs
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        
        if thickness < 0:  # Filled
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    def _add_text_with_background(self, frame, text, position, font_scale=0.6, 
                                   color=(255, 255, 255), bg_color=(0, 0, 0), 
                                   thickness=2, padding=5):
        """
        Add text with a background rectangle for better readability.
        
        Args:
            frame: Frame to draw on
            text: Text to display
            position: Text position (x, y)
            font_scale: Font size multiplier
            color: Text color (BGR)
            bg_color: Background color (BGR)
            thickness: Text thickness
            padding: Padding around text
        """
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.FONT, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(frame, text, (x, y), self.FONT, font_scale, color, thickness)
    
    def draw_status_panel(self, frame, analysis_result, statistics=None):
        """
        Draw a professional status panel showing posture metrics.
        
        Args:
            frame: Frame to draw on
            analysis_result: Result from PostureAnalyzer.analyze_posture()
            statistics: Optional session statistics
        """
        # Intentionally left blank: status and detailed metrics are shown in the
        # bottom GUI panel (Tkinter). To keep the camera feed clean this function
        # no longer draws any large status overlays on the frame.
        return
    
    def draw_angle_indicator(self, frame, keypoints, analysis_result):
        """
        Draw visual angle indicator showing neck alignment.
        
        Args:
            frame: Frame to draw on
            keypoints: Body keypoints dictionary
            analysis_result: Analysis result with neck angle
        """
        head = keypoints.get('head')
        neck = keypoints.get('neck')
        
        if None in [head, neck]:
            return
        
        # Draw angle arc
        neck_angle = analysis_result.get('neck_angle')
        if neck_angle is not None:
            # Calculate angle visualization
            radius = 50
            
            # Draw reference line (ideal posture)
            ref_angle = -90  # Straight vertical
            ref_end_x = int(neck[0] + radius * np.cos(np.radians(ref_angle)))
            ref_end_y = int(neck[1] + radius * np.sin(np.radians(ref_angle)))
            cv2.line(frame, neck, (ref_end_x, ref_end_y), (100, 100, 100), 1, cv2.LINE_AA)
            
            # Draw actual angle line
            # Calculate direction from neck to head
            dx = head[0] - neck[0]
            dy = head[1] - neck[1]
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            actual_end_x = int(neck[0] + radius * np.cos(angle_rad))
            actual_end_y = int(neck[1] + radius * np.sin(angle_rad))
            
            # Color based on posture quality
            line_color = self.COLOR_GOOD if neck_angle >= 150 else self.COLOR_POOR
            cv2.line(frame, neck, (actual_end_x, actual_end_y), line_color, 2, cv2.LINE_AA)
            
            # Draw arc between lines
            # This is simplified - a full arc would require more complex calculation
            cv2.ellipse(frame, neck, (radius, radius), 0, 
                       min(ref_angle, angle_deg), max(ref_angle, angle_deg),
                       line_color, 2)
    
    def draw_posture_guide(self, frame):
        """
        Draw a small posture guide in the corner showing ideal posture.
        
        Args:
            frame: Frame to draw on
        """
        # Draw in bottom-right corner
        guide_x = self.frame_width - 150
        guide_y = self.frame_height - 120
        guide_width = 130
        guide_height = 100
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (guide_x, guide_y), 
                     (guide_x + guide_width, guide_y + guide_height),
                     self.COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Ideal Posture:", (guide_x + 5, guide_y + 20),
                   self.FONT, 0.4, self.COLOR_INFO, 1)
        
        # Draw simple stick figure showing good posture
        stick_x = guide_x + 35
        stick_y_base = guide_y + 85
        
        # Head
        cv2.circle(frame, (stick_x, stick_y_base - 60), 10, self.COLOR_GOOD, 2)
        # Neck
        cv2.line(frame, (stick_x, stick_y_base - 50), (stick_x, stick_y_base - 35),
                self.COLOR_GOOD, 2)
        # Shoulders
        cv2.line(frame, (stick_x - 15, stick_y_base - 35), (stick_x + 15, stick_y_base - 35),
                self.COLOR_GOOD, 2)
        # Spine
        cv2.line(frame, (stick_x, stick_y_base - 35), (stick_x, stick_y_base),
                self.COLOR_GOOD, 2)
        
        # Labels
        cv2.putText(frame, "Straight", (guide_x + 65, guide_y + 40),
                   self.FONT, 0.35, self.COLOR_GOOD, 1)
        cv2.putText(frame, "Aligned", (guide_x + 65, guide_y + 60),
                   self.FONT, 0.35, self.COLOR_GOOD, 1)
    
    def draw_no_person_message(self, frame):
        """
        Draw a helpful message when no person is detected.
        
        Args:
            frame: Frame to draw on
        """
        message = "Please position yourself in front of the camera"
        sub_message = "Ensure your face and upper body are visible"
        
        # Calculate center position
        (text_width, text_height), _ = cv2.getTextSize(
            message, self.FONT, 0.8, 2
        )
        
        x = (self.frame_width - text_width) // 2
        y = self.frame_height // 2
        
        # Draw main message
        self._add_text_with_background(frame, message, (x, y), 
                                      font_scale=0.8, color=self.COLOR_WARNING,
                                      thickness=2, padding=10)
        
        # Draw sub-message
        (sub_width, _), _ = cv2.getTextSize(sub_message, self.FONT, 0.6, 1)
        sub_x = (self.frame_width - sub_width) // 2
        self._add_text_with_background(frame, sub_message, (sub_x, y + 40),
                                      font_scale=0.6, color=self.COLOR_INFO,
                                      thickness=1, padding=8)
    
    def draw_skeleton(self, frame, keypoints):
        """
        Draw skeletal overlay showing detected keypoints and connections.
        
        Args:
            frame: Input frame to draw on
            keypoints: Dictionary of keypoints from estimate_keypoints()
            
        Returns:
            Frame with skeleton overlay
        """
        if not keypoints:
            return frame
        
        # Define skeleton connections
        connections = [
            ('head', 'neck'),
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
                    cv2.line(frame, start, end, (0, 255, 255), 3)
        
        # Draw keypoints (joints)
        for key, point in keypoints.items():
            if point is not None:
                x, y = point
                # Draw joint circle
                cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
                
                # Label keypoint
                label = key.replace('_', ' ').title()
                cv2.putText(frame, label, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
