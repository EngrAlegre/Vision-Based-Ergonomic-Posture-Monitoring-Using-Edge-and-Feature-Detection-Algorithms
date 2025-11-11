import tkinter as tk
from tkinter import ttk
import time
import cv2
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False


class PostureApp:
    """Tkinter GUI wrapper for the posture monitoring system.

    This class expects a fully initialized system object that exposes the
    same components used in `main.py` (camera, person_detector, posture_detector,
    analyzer, visualizer, alert_system, logger).
    """

    def __init__(self, system):
        if not HAS_PIL:
            raise RuntimeError("Pillow (PIL) is required for the Tk GUI. Install with: pip install pillow")

        self.system = system
        self.root = tk.Tk()
        self.root.title("Vision-Based Ergonomic Posture Monitoring")
        self.root.configure(bg="#efefef")

        # Video frame area
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(padx=12, pady=(12, 6))

        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        # Bottom info panel
        self.bottom_panel = ttk.Frame(self.root)
        self.bottom_panel.pack(fill=tk.X, padx=12, pady=(6, 12))

        # Status label with dynamic feedback
        self.status_var = tk.StringVar(value="Status: --")
        self.status_label = tk.Label(self.bottom_panel, textvariable=self.status_var, fg="#1a8f2a",
                                     font=("Helvetica", 14, "bold"), bg="#efefef")
        self.status_label.pack(pady=(4, 2))

        # Percentages indicator (below status)
        self.percent_var = tk.StringVar(value="Good: --%   |   Poor: --%")
        self.percent_label = tk.Label(self.bottom_panel, textvariable=self.percent_var, fg="#1a8f2a",
                                      font=("Helvetica", 12, "bold"), bg="#efefef")
        self.percent_label.pack(pady=(0, 6))


        # Angle and percentages
        self.angle_var = tk.StringVar(value="Angle: --°")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.score_var = tk.StringVar(value="Score: --/100")

        info_frame = ttk.Frame(self.bottom_panel)
        info_frame.pack()

        self.angle_label = tk.Label(info_frame, textvariable=self.angle_var, font=("Helvetica", 12), bg="#efefef")
        self.angle_label.grid(row=0, column=0, padx=8)

        self.fps_label = tk.Label(info_frame, textvariable=self.fps_var, font=("Helvetica", 12), bg="#efefef")
        self.fps_label.grid(row=0, column=1, padx=8)
        
        self.score_label = tk.Label(info_frame, textvariable=self.score_var, font=("Helvetica", 12, "bold"), bg="#efefef")
        self.score_label.grid(row=0, column=2, padx=8)

        # Small issues line
        self.issues_var = tk.StringVar(value="")
        self.issues_label = tk.Label(self.bottom_panel, textvariable=self.issues_var, font=("Helvetica", 10), fg="#c45a00", bg="#efefef")
        self.issues_label.pack(pady=(6, 0))

        # Keep reference to PhotoImage to avoid garbage collection
        self._photo = None

        # Timing
        self.last_time = time.time()

        # Bind close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Flag to stop loop
        self._running = False

    def _on_close(self):
        self._running = False
        # Let main cleanup handle camera release and logger
        try:
            self.system.cleanup()
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the Tk mainloop and the update loop."""
        self._running = True
        self._update_loop()
        self.root.mainloop()

    def _update_loop(self):
        if not self._running:
            return

        try:
            frame = self.system.camera.get_frame()
            
            # Validate frame
            if frame is None or frame.size == 0:
                print("[GUI] Received invalid frame, skipping...")
                self.root.after(15, self._update_loop)
                return
            
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"[GUI] Frame has unexpected shape: {frame.shape}")
                self.root.after(15, self._update_loop)
                return

            # Run existing detection/analysis pipeline (mirrors main loop)
            frame, person_found, face_roi, body_roi = self.system.person_detector.detect(frame)

            analysis_result = {}
            statistics = {}

            if person_found:
                keypoints = self.system.posture_detector.estimate_keypoints(frame, face_roi, body_roi)
                frame = self.system.posture_detector.draw_skeleton(frame, keypoints)
                analysis_result = self.system.analyzer.analyze_posture(keypoints)
                statistics = self.system.analyzer.get_statistics()

                # Draw overlays using the visualizer
                self.system.visualizer.draw_status_panel(frame, analysis_result, statistics)
                self.system.visualizer.draw_angle_indicator(frame, keypoints, analysis_result)
                self.system.visualizer.draw_posture_guide(frame)

                # Alerts/logging
                if analysis_result.get('is_poor_posture', False) or analysis_result.get('is_poor', False):
                    self.system.alert_system.trigger_alert(analysis_result)
                # Log periodically
                # (we don't maintain frame_count here; leave logging to system where needed)
            else:
                self.system.visualizer.draw_no_person_message(frame)

            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time + 1e-6)
            self.last_time = current_time

            # ===== UPDATE GUI WITH SYNCED VISUAL FEEDBACK =====
            
            # Update status based on line quality (PRIMARY) + score (SECONDARY)
            line_quality = analysis_result.get('line_quality')  # Can be None
            score = analysis_result.get('score', 100)
            
            # Determine status color and text based on current posture
            status_color = "#1a8f2a"  # Green default
            if line_quality == 'poor' or score < 70:
                status_text = "POOR POSTURE"
                status_color = "#c40000"  # Red
            elif line_quality == 'warning' or score < 85:
                status_text = "WARNING"
                status_color = "#c45a00"  # Orange
            elif line_quality == 'calibrating':
                status_text = "CALIBRATING"
                status_color = "#DAA520"  # Gold
            else:
                status_text = "GOOD POSTURE"
                status_color = "#1a8f2a"  # Green
            
            # Show just status text
            self.status_var.set(f"Status: {status_text}")
            self.status_label.config(fg=status_color)
            
            # Show percentages below status
            good_pct = statistics.get('good_pct') if statistics else None
            poor_pct = statistics.get('poor_pct') if statistics else None
            if good_pct is not None and poor_pct is not None:
                self.percent_var.set(f"Good: {good_pct:.1f}% | Poor: {poor_pct:.1f}%")
                # Match status color for percentages too
                self.percent_label.config(fg=status_color)
            else:
                self.percent_var.set("Good: --%   |   Poor: --%")
                self.percent_label.config(fg="#808080")
            

            if score is not None:
                self.score_var.set(f"Score: {score}/100")
            else:
                self.score_var.set("Score: --/100")

            # Update line metrics from dynamic analysis
            neck_angle = analysis_result.get('neck_angle')
            if neck_angle is not None:
                self.angle_var.set(f"Angle: {neck_angle:.1f}°")
            else:
                self.angle_var.set("Angle: --°")

            self.fps_var.set(f"FPS: {fps:.1f}")

            # Issues
            issues = analysis_result.get('issues', [])
            if issues:
                self.issues_var.set("Issues: " + ", ".join(issues[:3]))
            else:
                self.issues_var.set("")

            # Convert for Tkinter
            try:
                if frame is None or frame.size == 0:
                    raise ValueError("Frame is invalid or empty")
                
                # Ensure frame is BGR
                if len(frame.shape) != 3:
                    raise ValueError(f"Frame has invalid shape: {frame.shape}")
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self._photo = imgtk
                self.canvas.configure(image=imgtk)
            except Exception as frame_error:
                print(f"[GUI] Frame conversion error: {frame_error}")
                # Create a placeholder image to show something
                placeholder = Image.new('RGB', (640, 480), color='black')
                imgtk = ImageTk.PhotoImage(image=placeholder)
                self._photo = imgtk
                self.canvas.configure(image=imgtk)

        except Exception as e:
            # swallow errors to keep UI responsive, print for debugging
            import traceback
            print(f"[GUI] Error in update loop: {e}")
            traceback.print_exc()

        # Schedule next frame (use small delay to keep UI snappy)
        self.root.after(15, self._update_loop)
