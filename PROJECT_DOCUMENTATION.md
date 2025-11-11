# Vision-Based Ergonomic Posture Monitoring — Project Documentation

This document consolidates the architecture, tools and techniques, file-by-file responsibilities, and the full list of OpenCV APIs used in the `Posture Detection` project. Paste it into your delivery or keep it as a reference.

---

## 1. High-level architecture

- `main.py` — orchestrator. Initializes subsystems: camera, detectors, analyzer, visualizer, alert system, logger. Runs either the Tk GUI (`tk_gui.PostureApp`) or the OpenCV display loop.
- `camera_module.Camera` — captures frames from webcam and performs optional preprocessing.
- `person_detector.PersonDetector` — detects face and upper-body ROIs (Haar cascades) and applies smoothing.
- `basic_posture_detector.BasicPostureDetector` — estimates keypoints (head, neck, shoulders, torso) and draws skeleton lines using classical CV techniques.
- `posture_analyzer.EnhancedPostureAnalyzer` — computes neck angles, forward-head displacement, shoulder alignment; assigns a posture score and session statistics.
- `visualizer.PostureVisualizer` — draws overlays (skeleton, guides, angle indicators). The large status panel was moved out of the frame so the camera feed stays clean.
- `tk_gui.PostureApp` — optional Tkinter GUI that shows the video feed and a bottom information panel (Status, Score, Angle, FPS, Good/Poor % and Issues). Depends on Pillow.
- `alert_system.AlertSystem` — cross-platform audio alerts with cooldown.
- `data_logger.DataLogger` — buffered CSV logging of posture measurements.


## 2. Tools & libraries used (non-OpenCV)

- Python 3.11+ (your environment uses Python 3.13)
- NumPy — numeric arrays and vector math
- Pillow (PIL) — convert OpenCV frames to images for Tkinter display (Image.fromarray + ImageTk)
- CSV / datetime / os — logging and file I/O utilities
- platform / winsound / shell audio calls — cross-platform audio alerts (Windows/macOS/Linux)
- collections.deque — temporal smoothing buffers


## 3. Techniques & algorithms implemented

- Haar cascades for face and upper-body detection (fast classical approach).
- Temporal smoothing of bounding boxes and keypoints (exponential smoothing / moving-average buffers) to reduce jitter.
- Skin color segmentation in YCrCb (cv2.inRange) to help localize exposed skin.
- Edge detection (Canny) + morphological ops + contour filtering to find torso-like shapes.
- Heuristic geometric keypoint estimation (head bottom-center, neck offset, shoulder positions relative to head and body ROI).
- Angle computation using vector dot-product (neck angle via three points).
- Rule-based scoring (0–100) combining neck angle, forward-head displacement, shoulder tilt, and centering.
- Buffered CSV logging to reduce disk I/O overhead. Alert cooldown to prevent frequent audio beeps.


## 4. File-by-file description (all `.py` files in `Posture Detection` folder)

- `main.py`
  - Entry point and orchestrator. Instantiates camera, detectors, analyzer, visualizer, alert system, and logger. Runs the main capture loop or `tk_gui.PostureApp` and handles cleanup.

- `camera_module.py`
  - Encapsulates webcam access (cv2.VideoCapture), frame acquisition, basic configuration (frame size and FPS), and cleanup.

- `person_detector.py`
  - Uses Haar cascade classifiers (`haarcascade_frontalface_default.xml` and `haarcascade_upperbody.xml`) to detect face and torso ROIs. Applies temporal smoothing to bounding boxes and draws labeled rectangles on frames.

- `basic_posture_detector.py`
  - Estimates keypoints (head, neck, left/right shoulders, torso center) using geometric heuristics and contour analysis. Performs skin detection and contour-based body detection as a fallback. Draws the skeleton overlay (lines, joint circles).

- `posture_analyzer.py`
  - Computes posture metrics: neck angle, forward-head displacement, shoulder tilt, overall alignment. Produces a `analysis_result` dictionary (including `score` and `issues`) and maintains session statistics.

- `visualizer.py`
  - Draws UX overlays: skeleton, angle indicator, posture guide, and `draw_no_person_message`. Previously drew a large semi-transparent status panel; that was intentionally moved out of the video frame so the GUI bottom panel displays detailed status instead. You can re-enable in-frame panels if desired.

- `tk_gui.py`
  - Tkinter GUI wrapper that displays live frames (via Pillow ImageTk) and a bottom panel with `Status`, `Score`, `Angle`, `FPS`, `Good/Poor %`, and `Issues`. The bottom `Status` label toggles green/red depending on posture quality.

- `alert_system.py`
  - Plays audio alerts in a cross-platform way (Windows `winsound`, macOS `afplay`, Linux `paplay`), with an alert cooldown. Returns textual messages for the UI.

- `data_logger.py`
  - Buffered CSV logger that writes posture samples periodically and provides helpers to read recent entries.

- `__pycache__/` — Python bytecode cache (auto-generated; safe to ignore or delete).

- `ENHANCEMENT (MAY REMOVE)/` — optional folder for enhancements (inspect before deleting).


## 5. OpenCV APIs and functions used (comprehensive list)

Capture & camera I/O
- `cv2.VideoCapture(...)`
- `cv2.CAP_PROP_FRAME_WIDTH`, `cv2.CAP_PROP_FRAME_HEIGHT`, `cv2.CAP_PROP_FPS`
- `cap.read()`, `cap.release()`
- `cv2.destroyAllWindows()`

Windowing / display / input
- `cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)`
- `cv2.imshow(window_name, frame)`
- `cv2.waitKey(ms)`

Color conversions
- `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
- `cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)`
- `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` (for Pillow conversion)

Filters & denoising
- `cv2.fastNlMeansDenoisingColored(...)` (optional/commented)
- `cv2.bilateralFilter(...)`
- `cv2.GaussianBlur(...)`

Thresholding / color detection / morphology
- `cv2.inRange(image, lower, upper)`
- `cv2.morphologyEx(..., cv2.MORPH_OPEN / cv2.MORPH_CLOSE)`
- `cv2.getStructuringElement(cv2.MORPH_ELLIPSE / cv2.MORPH_RECT, size)`

Edge detection & contours
- `cv2.Canny(filtered, threshold1, threshold2)`
- `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
- `cv2.contourArea(contour)`, `cv2.boundingRect(contour)`

Haar cascade detection
- `cv2.CascadeClassifier(...)` using `cv2.data.haarcascades` path
- `cascade.detectMultiScale(...)`

Drawing primitives
- `cv2.rectangle(frame, pt1, pt2, color, thickness)`
- `cv2.putText(frame, text, org, font, fontScale, color, thickness)`
- `cv2.line(frame, pt1, pt2, color, thickness)`
- `cv2.circle(frame, center, radius, color, thickness)`
- `cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, color, thickness)`
- `cv2.addWeighted(overlay, alpha, frame, beta, gamma)`
- `cv2.getTextSize(text, font, fontScale, thickness)`

Constants & flags
- `cv2.FONT_HERSHEY_SIMPLEX`, `cv2.FONT_HERSHEY_DUPLEX`
- `cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`

Utilities for display in Tk
- Convert BGR → RGB then `PIL.Image.fromarray` and `ImageTk.PhotoImage` in `tk_gui.py`.


## 6. Safety / edge cases & notes

- Haar cascades are lightweight but less robust than landmark-based approaches (e.g., MediaPipe). Consider switching to MediaPipe Pose/Face if you need more accuracy.
- Low lighting, occlusions, and unusual camera angles negatively affect Haar + contour heuristics.
- CPU usage can be significant at full FPS; consider downsampling or skipping frames to reduce load.
- `__pycache__` is safe to delete. Inspect `ENHANCEMENT (MAY REMOVE)` before deleting.


## 7. Suggested next improvements (optional)

- Add `requirements.txt` with pinned versions (`opencv-python`, `numpy`, `pillow`) — I can create it.
- Add `README.md` with run instructions, dependency install commands, and troubleshooting steps — I can create it.
- Add unit tests for deterministic parts (`calculate_angle`, scoring logic in `posture_analyzer`) — quick wins for reliability.
- Offer a MediaPipe-based implementation for improved landmark detection with a graceful fallback to current heuristics.
- Add an OpenCV-only bottom status panel for users without Tk/Pillow.

