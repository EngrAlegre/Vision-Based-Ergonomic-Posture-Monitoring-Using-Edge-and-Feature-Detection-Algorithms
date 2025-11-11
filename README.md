# Vision-Based Ergonomic Posture Monitoring Using Edge and Feature Detection Algorithms

**Project Team:**
- Alegre, Jhon Isaac
- Corpuz, Micki Laurren B.
- De Guia, Lloyd James
- Deniega, Alexis Neil
- Manuel, Hazel Aillson T.

---

## Overview

This is a real-time vision-based posture monitoring system that uses only traditional computer vision techniques to detect and analyze human posture from a live webcam feed. The system identifies the user's head, neck, and shoulder alignment using geometric estimations and basic image processing algorithms such as edge detection, contour extraction, and Haar-based face detection.

The implementation relies solely on **OpenCV and NumPy**, requiring no machine learning or deep learning models. This approach demonstrates the feasibility of ergonomic monitoring using classical algorithms.

---

## Key Features

- **Real-time posture detection** using webcam feed
- **Traditional computer vision** - no machine learning required
- **Geometric keypoint estimation** for head, neck, and shoulders
- **Dynamic visual feedback** with color-coded posture indicators
- **Audio alerts** for poor posture detection
- **Live GUI** showing metrics and session statistics
- **Cross-platform** support (Windows, macOS, Linux)

---

## System Requirements

- Python 3.11 or higher (tested on Python 3.13)
- Webcam (built-in or external)
- Windows, macOS, or Linux operating system

---

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt ('you can also install it inside of venv')
```

### 2. Verify Installation

```bash
python -c "import cv2, numpy, PIL; print('All dependencies installed successfully')"
```

---

## Quick Start

### Run the Application

```bash
python main.py
```

### Position Yourself

- Sit at normal distance (60-90 cm) from the camera
- Ensure your face and upper body are visible
- Good lighting helps detection accuracy

### Understanding the Feedback

**Good Posture (Green):**
- Head-neck line remains vertical and long
- Shoulders are level
- Body is centered

**Poor Posture (Red):**
- Head-neck line shortens (forward head posture)
- Line bends significantly (neck tilt/misalignment)
- Shoulders are uneven

---

## How It Works

### 1. Face Detection
Uses OpenCV's Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to detect the user's face, establishing the reference point for head position.

### 2. Body Segmentation
Applies **Canny Edge Detection** and **contour extraction** to identify the upper-body contour and isolate the torso from the background.

### 3. Keypoint Estimation
Estimates anatomical landmarks (head, neck, shoulders, torso center) using **geometric projection** - no pose estimation models required.

### 4. Posture Analysis
Computes:
- **Euclidean distance** between head and neck (shortens during slouching)
- **Angular deviation** from vertical (increases during neck tilting)
- **Posture score** (0-100) based on multiple metrics

### 5. Visual Feedback
Renders a **dynamic overlay** on the video feed:
- **Green skeleton/lines** = Good posture
- **Red skeleton/lines** = Poor posture
- Real-time angle and score display

### 6. User Interface
**Tkinter GUI** displays:
- Live camera feed with posture overlay
- Status (Good/Poor)
- Neck angle measurement
- Posture score (0-100)
- Session statistics (Good % / Poor %)
- Frame rate (FPS)

---

## Technical Details

### Computer Vision Techniques Used

| Technique | Purpose | Module |
|-----------|---------|--------|
| **Haar Cascade** | Face detection | `cv2.CascadeClassifier` |
| **Canny Edge Detection** | Body contour extraction | `cv2.Canny` |
| **Contour Analysis** | Body segmentation | `cv2.findContours` |
| **Morphological Operations** | Noise reduction | `cv2.morphologyEx` |
| **Color Space Conversion** | Grayscale processing | `cv2.cvtColor` |
| **Geometric Projection** | Keypoint estimation | NumPy calculations |
| **Temporal Smoothing** | Reduce jitter | Moving average buffers |

### Posture Classification Rules

**Poor posture is detected when:**
- Neck angle < 150° (forward tilt)
- Head displacement > 30 pixels forward
- Shoulder height difference > 15 pixels
- Overall score < 70/100

---

## Project Structure

```
Final project/
├── main.py                      # Application launcher and orchestrator
├── camera_module.py             # Webcam capture and preprocessing
├── person_detector.py           # Face detection (Haar cascades)
├── basic_posture_detector.py    # Keypoint estimation using geometry
├── posture_analyzer.py          # Posture metrics and scoring
├── visualizer.py                # Visual overlays and feedback
├── tk_gui.py                    # Tkinter GUI interface
├── alert_system.py              # Audio alert system
└── requirements.txt             # Python dependencies
```

---

## Module Descriptions

### `main.py`
Application entry point. Initializes all subsystems (camera, detectors, analyzer, visualizer, alerts) and runs the main processing loop.

### `camera_module.py`
Handles webcam access and frame capture with configurable resolution and FPS settings.

### `person_detector.py`
Detects face and upper-body regions using Haar cascade classifiers with temporal smoothing for stable bounding boxes.

### `basic_posture_detector.py`
Estimates keypoints (head, neck, shoulders, torso) using geometric heuristics, edge detection, and contour analysis. Draws skeleton overlay.

### `posture_analyzer.py`
Computes posture metrics (neck angle, forward head displacement, shoulder alignment) and assigns a quality score (0-100).

### `visualizer.py`
Renders visual feedback: skeleton lines, angle indicators, posture guide, and status messages.

### `tk_gui.py`
Provides a graphical user interface with live video feed and bottom information panel showing metrics and statistics.

### `alert_system.py`
Cross-platform audio alert system that triggers sound notifications for poor posture with cooldown mechanism.

---

## Troubleshooting

### Camera Not Detected
- Check camera permissions in system settings
- Ensure no other application is using the camera
- Try changing `cam_index` in `main.py` (0, 1, or 2)

### Face Not Detected
- Improve lighting (face the light source)
- Move closer to camera (60-90 cm optimal)
- Ensure face is clearly visible and not obscured

### Poor Detection Accuracy
- Adjust lighting to reduce shadows
- Keep upper body fully in frame
- Avoid patterned clothing that disrupts contours
- Maintain consistent distance from camera

### Low Frame Rate
- Close other applications to free CPU
- Reduce camera resolution in `camera_module.py`
- Disable unnecessary visual overlays

---

## System Performance

- **Frame Rate:** 12-30 FPS (depending on hardware)
- **Detection Range:** 60-150 cm from camera
- **Lighting Requirements:** Indoor lighting or natural light
- **CPU Usage:** Low to moderate (no GPU required)

---

## Addressing Musculoskeletal Disorders

This system addresses a critical health issue in the Philippines:

- **39%** of occupational diseases are attributed to back pain
- **MSDs increased by 75%** between 2013-2015 (45,572 → 78,716 cases)
- **54,551** occupational diseases recorded in 2019
- **80%** of Filipino adults experience back pain at some point
- **12.1%** of cases involve neck-shoulder pain

By providing real-time feedback, this system helps users maintain proper posture and reduce the risk of developing musculoskeletal disorders.

---

## Advantages of This  (For Project Only)

✅ **No Machine Learning Required** - Uses only classical CV algorithms  
✅ **Lightweight & Fast** - Runs on any computer with a webcam  
✅ **Explainable** - All calculations are geometric and deterministic  
✅ **Privacy-Friendly** - No data sent to external servers  
✅ **Low Computational Cost** - No GPU required  
✅ **Offline Capable** - Works without internet connection  

---

## Limitations

- Accuracy depends on lighting quality and contrast
- Requires proper camera positioning and calibration
- May be affected by patterned clothing or cluttered backgrounds
- Designed for frontal sitting postures (not suitable for side views)

---

## Future Enhancements

- Adaptive calibration for multiple users
- Lightweight 3D modeling using dual-camera depth approximation
- Mobile and embedded platform support (e.g., Raspberry Pi)
- Historical posture tracking and analytics
- Multi-angle detection support

