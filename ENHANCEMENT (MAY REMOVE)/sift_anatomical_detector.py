"""
SIFT-Based Anatomical Landmark Detector (Module 7: Feature Extraction)
========================================================================

This module uses Scale-Invariant Feature Transform (SIFT) to detect
distinctive anatomical features that can supplement geometric posture analysis.

SIFT is a classical CV technique (no ML) that detects and describes
local features that are invariant to scale, rotation, and illumination.
"""

import cv2
import numpy as np
from collections import defaultdict

class SIFTAnatomicalDetector:
    """
    Uses SIFT keypoints to identify distinctive anatomical features.
    Supplements geometric estimation with feature-based detection.
    
    Module 7: Feature Extraction - SIFT
    """
    
    def __init__(self):
        """Initialize SIFT detector with parameters tuned for body features."""
        try:
            # Try to create SIFT (may require opencv-contrib-python)
            self.sift = cv2.SIFT_create(
                nfeatures=100,  # Limit keypoints for performance
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
            self.sift_available = True
        except AttributeError:
            # Fallback to ORB if SIFT not available
            print("⚠️ SIFT not available, using ORB as fallback")
            self.sift = cv2.ORB_create(nfeatures=100)
            self.sift_available = False
        
        # Keypoint clustering parameters
        self.cluster_threshold = 50  # pixels
        
    def detect_keypoints(self, frame, roi=None):
        """
        Detect SIFT keypoints in frame or ROI.
        
        Args:
            frame: Input BGR frame
            roi: Optional ROI (x, y, w, h) to focus detection
            
        Returns:
            tuple: (keypoints, descriptors, annotated_frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on ROI if provided
        if roi is not None:
            x, y, w, h = roi
            gray_roi = gray[y:y+h, x:x+w]
            keypoints, descriptors = self.sift.detectAndCompute(gray_roi, None)
            
            # Adjust keypoint coordinates to full frame
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        else:
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def cluster_keypoints_spatial(self, keypoints):
        """
        Cluster keypoints spatially to identify body regions.
        
        Groups keypoints that are close together, which often
        correspond to distinctive anatomical features (joints, etc).
        
        Args:
            keypoints: List of cv2.KeyPoint objects
            
        Returns:
            Dictionary mapping region names to keypoint clusters
        """
        if len(keypoints) == 0:
            return {}
        
        # Extract keypoint coordinates
        points = np.array([kp.pt for kp in keypoints])
        
        # Simple spatial clustering (group nearby points)
        clusters = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
            
            # Find all points within threshold distance
            cluster = [i]
            for j, other_point in enumerate(points):
                if j != i and j not in used:
                    distance = np.linalg.norm(point - other_point)
                    if distance < self.cluster_threshold:
                        cluster.append(j)
                        used.add(j)
            
            used.add(i)
            
            if len(cluster) >= 3:  # Minimum cluster size
                clusters.append(cluster)
        
        # Compute cluster centers
        cluster_centers = []
        for cluster in clusters:
            cluster_points = points[cluster]
            center = np.mean(cluster_points, axis=0)
            size = len(cluster)
            cluster_centers.append({
                'center': tuple(center.astype(int)),
                'size': size,
                'keypoint_indices': cluster
            })
        
        return cluster_centers
    
    def estimate_anatomical_points(self, frame, person_roi, keypoints):
        """
        Estimate anatomical landmark positions using SIFT keypoints.
        
        Identifies regions with high keypoint density that correspond
        to body joints and features.
        
        Args:
            frame: Input frame
            person_roi: Person bounding box (x, y, w, h)
            keypoints: List of SIFT keypoints
            
        Returns:
            Dictionary of estimated anatomical points
        """
        if person_roi is None or len(keypoints) == 0:
            return {}
        
        x, y, w, h = person_roi
        
        # Cluster keypoints spatially
        clusters = self.cluster_keypoints_spatial(keypoints)
        
        if len(clusters) == 0:
            return {}
        
        # Sort clusters by vertical position (top to bottom)
        clusters.sort(key=lambda c: c['center'][1])
        
        anatomical_points = {}
        
        # Heuristic: Top cluster likely near head/shoulders
        if len(clusters) >= 1:
            top_cluster = clusters[0]
            anatomical_points['head_region'] = top_cluster['center']
        
        # Middle clusters likely near torso/arms
        if len(clusters) >= 2:
            mid_cluster = clusters[len(clusters) // 2]
            anatomical_points['torso_region'] = mid_cluster['center']
        
        # Bottom cluster likely near hips/legs
        if len(clusters) >= 3:
            bottom_cluster = clusters[-1]
            anatomical_points['lower_region'] = bottom_cluster['center']
        
        return anatomical_points
    
    def draw_keypoints_and_clusters(self, frame, keypoints, clusters=None):
        """
        Visualize SIFT keypoints and their spatial clusters.
        
        Args:
            frame: Input frame
            keypoints: List of keypoints
            clusters: Optional cluster information
            
        Returns:
            Annotated frame
        """
        # Draw individual keypoints (small)
        frame = cv2.drawKeypoints(
            frame, keypoints, None,
            color=(0, 255, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Draw cluster centers (large)
        if clusters:
            for i, cluster in enumerate(clusters):
                center = cluster['center']
                size = cluster['size']
                
                # Draw cluster center
                cv2.circle(frame, center, 15, (255, 0, 255), 2)
                cv2.circle(frame, center, 5, (255, 0, 255), -1)
                
                # Label cluster
                label = f"C{i+1} ({size})"
                cv2.putText(frame, label, (center[0] + 20, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return frame


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demo_sift_detection():
    """
    Demonstration of SIFT-based anatomical feature detection.
    
    SIFT Benefits:
    - Detects distinctive features (corners, edges)
    - Scale and rotation invariant
    - Works well for textured regions (clothing patterns)
    - Can identify joint locations
    
    SIFT Limitations:
    - Doesn't work well on plain/smooth surfaces
    - Requires distinctive visual features
    - More keypoints on textured clothing than skin
    
    Integration Strategy:
    - Use SIFT to supplement geometric estimation
    - Cluster keypoints to identify body regions
    - Validate geometric estimates with SIFT clusters
    """
    print("=" * 60)
    print("SIFT Anatomical Feature Detection Demo (Module 7)")
    print("=" * 60)
    
    # Initialize detector
    detector = SIFTAnatomicalDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    print("\n✓ SIFT detector initialized")
    print("✓ Camera opened")
    print("\nPress 'Q' to quit")
    print("\nNote: SIFT works best with patterned/textured clothing!\n")
    
    # For person detection (use simple background subtraction)
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simple person ROI estimation (you can replace with Haar/HOG)
        fg_mask = backSub.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        person_roi = None
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5000:
                person_roi = cv2.boundingRect(largest)
                x, y, w, h = person_roi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect SIFT keypoints
        keypoints, descriptors = detector.detect_keypoints(frame, person_roi)
        
        # Cluster keypoints
        clusters = detector.cluster_keypoints_spatial(keypoints)
        
        # Estimate anatomical points
        anatomical_pts = detector.estimate_anatomical_points(
            frame, person_roi, keypoints
        )
        
        # Visualize
        frame = detector.draw_keypoints_and_clusters(frame, keypoints, clusters)
        
        # Draw anatomical point estimates
        for name, point in anatomical_pts.items():
            cv2.circle(frame, point, 8, (0, 0, 255), -1)
            cv2.putText(frame, name.replace('_', ' ').title(), 
                       (point[0] + 15, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Show stats
        cv2.putText(frame, f"Keypoints: {len(keypoints)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Clusters: {len(clusters)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("SIFT Anatomical Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_sift_detection()


# ============================================================================
# INTEGRATION WITH MAIN SYSTEM
# ============================================================================
"""
INTEGRATION STRATEGY: Hybrid Approach

Combine geometric estimation (fast, reliable) with SIFT features (detailed).

In main.py:

    # Geometric estimation (primary method)
    keypoints_geo = posture_detector.estimate_keypoints(frame, face_roi, body_roi)
    
    # SIFT features (validation/refinement)
    sift_detector = SIFTAnatomicalDetector()
    sift_kps, _ = sift_detector.detect_keypoints(frame, body_roi)
    clusters = sift_detector.cluster_keypoints_spatial(sift_kps)
    
    # Use SIFT clusters to validate geometric estimates
    # If SIFT finds a strong cluster near geometric shoulder estimate,
    # we have higher confidence in that estimate
    
WHEN TO USE:
- Geometric: Always (primary method)
- SIFT: When person wears patterned clothing (high keypoint density)
- Combined: Use SIFT to refine geometric estimates

BENEFITS:
- More robust keypoint estimation
- Can handle challenging poses
- Validates geometric assumptions
- Works better with textured clothing
