# 🎯 Improved Posture Detection System

## Overview
This is an **enhanced version** of your posture detection project that uses **ONLY basic computer vision techniques** - no machine learning, no MediaPipe, no deep learning models. The system detects and analyzes your sitting posture in real-time using geometric analysis and classical CV algorithms.

---

## 🚀 Key Improvements

### 1. ❌ **REMOVED MediaPipe (ML-Based)**
**What was changed:**
- Completely removed MediaPipe pose estimation library
- Removed all ML-based pose detection code

**Why it was changed:**
- MediaPipe uses deep learning models (violates your "no ML" constraint)
- You wanted only basic CV techniques as taught in coursework

**How it improves the project:**
- ✅ Now truly uses only basic computer vision
- ✅ More transparent and understandable algorithm
- ✅ Aligns with project requirements

**Replaced with:**
- `BasicPostureDetector` class using geometric body analysis
- Contour-based body segmentation
- Color-based skin detection (YCrCb color space)
- Mathematical estimation of body keypoints

---

### 2. 🎨 **Enhanced Visual Output**

Vision-Based Ergonomic Posture Monitoring
=========================================

A lightweight posture monitoring project that uses classical computer-vision techniques (OpenCV + NumPy) to estimate head/neck/shoulder alignment and report posture quality in real-time.

Quick start (Windows / PowerShell)
---------------------------------

1) Verify you are using the intended Python executable. Example (your environment):

```powershell
& C:/Users/hazel/AppData/Local/Microsoft/WindowsApps/python3.13.exe -V
```

2) Install dependencies from `requirements.txt` (installs OpenCV, NumPy, Pillow):

```powershell
& C:/Users/hazel/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install --user -r "c:/Users/hazel/Downloads/Personal Projects/Posture Detection/requirements.txt"
```

3) (Optional) Install MediaPipe for improved landmark detection (may require a matching Python version and available wheel):

```powershell
& C:/Users/hazel/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m pip install --user mediapipe
```

4) Run the app:

```powershell
& C:/Users/hazel/AppData/Local/Microsoft/WindowsApps/python3.13.exe "c:/Users/hazel/Downloads/Personal Projects/Posture Detection/main.py"
```

Notes
-----
- The app prefers the Tkinter GUI (`tk_gui.py`) when Pillow is available. If Pillow is not installed, it falls back to the OpenCV window.
- The GUI displays the camera feed and a bottom info panel (Status, Score, Angle, FPS, Good/Poor % and Issues). Visual overlays (skeleton/lines/angle indicator) are drawn on the frame; large textual status has been moved to the bottom panel for readability.
- If your webcam is not detected you will see a runtime error from `camera_module.Camera`. Check camera permissions and that no other app is blocking the webcam.

Project layout (key files)
--------------------------
- `main.py` — application entry and orchestrator.
- `camera_module.py` — webcam capture and preprocessing.
- `person_detector.py` — face and upper-body detection (Haar cascades).
- `basic_posture_detector.py` — keypoint estimation (head/neck/shoulders) and skeleton drawing.
- `posture_analyzer.py` — computes posture metrics and score.
- `visualizer.py` — draws overlays on frames (skeleton, arc indicators, guide).
- `tk_gui.py` — Tkinter GUI wrapper displaying frames and bottom status panel.
- `alert_system.py` — audio alert handler (cross-platform support).
- `data_logger.py` — buffered CSV logging of measurements.

Troubleshooting
---------------
- If frames are blank or detection fails: try better lighting and ensure your face and upper torso are visible to the camera.
- To force the OpenCV fallback (no Tk GUI), uninstall or temporarily rename Pillow so the Tk path is not available.
- To reduce CPU usage: reduce camera resolution in `camera_module.py` (e.g., 320x240) or skip frames in the main loop.

Next steps & improvements
-------------------------
- (Optional) Integrate MediaPipe for more accurate and robust landmark detection.
- Add a `requirements.txt` lock or pinned versions if needed for reproducible installs (this repository includes a basic `requirements.txt`).
- Add unit tests for deterministic functions in `posture_analyzer.py`.

License & credits
-----------------
- This project uses OpenCV and NumPy (BSD-like licenses). If you add external models or data, include their attribution as appropriate.

