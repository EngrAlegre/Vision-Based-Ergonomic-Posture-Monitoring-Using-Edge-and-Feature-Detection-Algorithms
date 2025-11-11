import cv2
import sys
import time

from camera_module import Camera
from person_detector import PersonDetector
from basic_posture_detector import BasicPostureDetector
from posture_analyzer import EnhancedPostureAnalyzer
from visualizer import PostureVisualizer
from alert_system import AlertSystem


class PostureMonitoringSystem:
    """Main system coordinator for posture monitoring."""
    
    def __init__(self):
        """Initialize all system components."""
        
        try:
            # Initialize camera
            self.camera = Camera(cam_index=0, width=640, height=480)
            print("[OK] Camera initialized")
            
            # Initialize detection modules
            self.person_detector = PersonDetector()
            print("[OK] Person detector loaded")
            
            self.posture_detector = BasicPostureDetector()
            print("[OK] Basic posture detector initialized")
            
            self.analyzer = EnhancedPostureAnalyzer()
            print("[OK] Posture analyzer ready")
            
            # Initialize visualization
            self.visualizer = PostureVisualizer(
                frame_width=640,
                frame_height=480
            )
            print("[OK] Visualizer configured")
            
            # Initialize alert and logging
            self.alert_system = AlertSystem(
                enable_sound=True,
                alert_cooldown=5.0
            )
            print("[OK] Alert system active")
            
            print("=" * 60)
            print("[OK] System ready! Press 'Q' to quit\n")
            
            # Performance tracking
            self.fps_buffer = []
            self.last_time = time.time()
            
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            sys.exit(1)
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        
        # Moving average for smooth FPS display
        self.fps_buffer.append(fps)
        if len(self.fps_buffer) > 10:
            self.fps_buffer.pop(0)
        
        return sum(self.fps_buffer) / len(self.fps_buffer)
    
    def run(self):
        """Main processing loop."""
        window_name = "✨ Improved Posture Monitoring System"

        # Prefer Tkinter GUI (if available). Falls back to OpenCV window if not.
        try:
            from tk_gui import PostureApp
            print("[OK] Launching Tkinter GUI...")
            app = PostureApp(self)
            app.run()
            return
        except Exception:
            # Fall back to OpenCV window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print("[OK] Starting video capture...")
            print("[INFO] Tips:")
            print("   - Sit naturally in front of the camera")
            print("   - Ensure your face and upper body are visible")
            print("   - Good lighting helps detection accuracy")
            print("\n[INFO] Processing...\n")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                frame = self.camera.get_frame()
                frame_count += 1
                
                # Detect person (face and body)
                frame, person_found, face_roi, body_roi = self.person_detector.detect(frame)
                
                if not person_found:
                    # No person detected - show helpful message
                    self.visualizer.draw_no_person_message(frame)
                else:
                    # Person detected - proceed with posture analysis
                    
                    # Estimate body keypoints using basic CV
                    keypoints = self.posture_detector.estimate_keypoints(
                        frame, face_roi, body_roi
                    )
                    
                    # Draw skeletal overlay
                    frame = self.posture_detector.draw_skeleton(frame, keypoints)
                    
                    # Analyze posture using multiple metrics
                    analysis_result = self.analyzer.analyze_posture(keypoints)
                    
                    # Get session statistics
                    statistics = self.analyzer.get_statistics()
                    
                    # Draw professional status panel
                    self.visualizer.draw_status_panel(
                        frame, analysis_result, statistics
                    )
                    
                    # Draw angle indicator
                    self.visualizer.draw_angle_indicator(
                        frame, keypoints, analysis_result
                    )
                    
                    # Draw posture guide
                    self.visualizer.draw_posture_guide(frame)
                    
                    # Trigger alerts if needed
                    if analysis_result.get('is_poor_posture', False):
                        alert_triggered = self.alert_system.trigger_alert(analysis_result)
                        if alert_triggered:
                            print(f"⚠️  Poor posture alert! Score: {analysis_result.get('score', 0)}/100")
                
                # Display FPS
                fps = self.calculate_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Shutting down...")
                    break
                elif key == ord('r') or key == ord('R'):
                    # Reset statistics
                    self.analyzer.reset_statistics()
                    print("🔄 Statistics reset")
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[INFO] Cleaning up...")
        
        # Get final statistics
        stats = self.analyzer.get_statistics()
        print(f"\n[SUMMARY] Session Summary:")
        print(f"   Total measurements: {stats.get('total', 0)}")
        print(f"   Good posture: {stats.get('good_pct', 0):.1f}%")
        print(f"   Poor posture: {stats.get('poor_pct', 0):.1f}%")
        
        # Release camera
        self.camera.release()
        print("[OK] Camera released")
        
        print("\n[OK] Thank you for using the Posture Monitoring System!")


def main():
    """Entry point."""
    print("\n" + "=" * 60)
    print("  Vision-Based Ergonomic Posture Monitoring Using Edge")
    print("         and Feature Detection Algorithms")
    print("=" * 60 + "\n")
    
    system = PostureMonitoringSystem()
    system.run()


if __name__ == "__main__":
    main()
