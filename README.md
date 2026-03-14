# 🤖 OpenCV Computer Vision Workshop

> **A hands-on beginner workshop on Computer Vision using Python, OpenCV, and YOLO.**

---

## 📋 Table of Contents

- [What is Computer Vision?](#-what-is-computer-vision)
- [What is OpenCV?](#-what-is-opencv)
- [How OpenCV Works Internally](#-how-opencv-works-internally)
- [Why Python?](#-why-python-for-opencv)
- [Environment Setup](#-environment-setup)
- [Understanding Digital Images](#-understanding-digital-images)
- [Core OpenCV Operations](#-core-opencv-operations)
- [Webcam Video Capture](#-webcam-video-capture)
- [Image Filters & Edge Detection](#-image-filters--edge-detection)
- [Color Detection](#-color-detection)
- [Face Detection](#-face-detection)
- [YOLO Object Detection](#-yolo-object-detection)
- [Workshop Activities](#-workshop-activities)
- [Learning Outcomes](#-learning-outcomes)

---

## 👁️ What is Computer Vision?

Computer Vision is a field of AI that enables computers to **interpret and understand visual information from the world**.

**Real-world examples:**
- 📱 Face unlock in smartphones
- 🚗 Self-driving cars
- 📷 Security surveillance
- 🏥 Medical image analysis
- 🕶️ Augmented reality filters

> **In simple words: Computer Vision = Teaching computers how to see.**

---

## 📦 What is OpenCV?

**OpenCV** stands for **Open Source Computer Vision Library**.

It provides thousands of functions for:

| Category | Examples |
|---|---|
| Image Processing | Resize, crop, blur, filter |
| Object Detection | Haar Cascades, HOG detectors |
| Face Detection | Frontal face, eye, smile |
| Motion Tracking | Optical flow |
| Camera Calibration | Lens distortion correction |

**Used in:** Robotics · Autonomous Vehicles · Medical Imaging · Industrial Automation · AI Research

---

## 🧠 How OpenCV Works Internally

OpenCV processes images using a **pipeline of operations** — like a factory assembly line where each stage transforms the image.

```
Camera Frame
     │
     ▼
Image Acquisition       ← "Opening your eyes"
     │
     ▼
Image Representation    ← Stored as matrix of numbers
     │
     ▼
Preprocessing           ← Resize, denoise, convert color
     │
     ▼
Feature Detection       ← Find edges, corners, keypoints
     │
     ▼
Object Detection        ← Haar Cascades / YOLO
     │
     ▼
Visualization           ← Display result
```

### Human Vision vs Computer Vision

```
Human:    Eye    → Brain      → Recognize Object
Computer: Camera → OpenCV/AI  → Object Recognition
```

> **Humans see using eyes and brain. Computers see using cameras and math.**

---

## 🐍 Why Python for OpenCV?

- ✅ Simple, readable syntax
- ✅ Rapid prototyping
- ✅ Huge AI/ML ecosystem (NumPy, TensorFlow, PyTorch)
- ✅ Strong community support

---

## ⚙️ Environment Setup

### 1. Install Python

Download from [python.org](https://www.python.org) and verify:

```bash
python --version
```

### 2. Install PyCharm IDE

Download [PyCharm Community Edition](https://www.jetbrains.com/pycharm/) (free).

**Setup steps:**
1. Open PyCharm → Click **New Project**
2. Select **Pure Python**
3. Create your project

### 3. Install Required Libraries

```bash
pip install opencv-python numpy matplotlib ultralytics
```

Or install one by one:

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install ultralytics
```

### 4. Verify Installation

Create `test_setup.py` and run it:

```python
import cv2
import numpy as np
from ultralytics import YOLO

print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)
print("YOLO Library Loaded Successfully")
```

✅ No errors = you're good to go!

### 5. Project Folder Structure

```
opencv-workshop/
├── images/
│   └── sample.jpg
├── scripts/
│   ├── image_display.py
│   ├── webcam.py
│   ├── color_detection.py
│   └── face_detection.py
├── models/
└── requirements.txt
```

---

## 🖼️ Understanding Digital Images

Images are stored as **arrays of numbers (pixels)**.

### Grayscale Image

Each pixel = brightness value (0–255)

```
120  100   90
200  210  180
 30   40   50
```

| Value | Meaning |
|---|---|
| `0` | Black |
| `255` | White |

### Color Image (BGR)

OpenCV uses **BGR** channel order (not RGB):

```python
[255, 0, 0]  → Blue
[0, 255, 0]  → Green
[0, 0, 255]  → Red
```

### Image Shape

```python
print(img.shape)
# Output: (720, 1280, 3)
```

| Value | Meaning |
|---|---|
| `720` | Height (pixels) |
| `1280` | Width (pixels) |
| `3` | Color Channels (BGR) |

---

## 🔧 Core OpenCV Operations

### Load & Display an Image

```python
import cv2

img = cv2.imread("sample.jpg")
cv2.imshow("Image Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

| Function | Purpose |
|---|---|
| `cv2.imread()` | Reads image from file |
| `cv2.imshow()` | Displays image in window |
| `cv2.waitKey(0)` | Waits for any key press |
| `cv2.destroyAllWindows()` | Closes all windows |

### Convert to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
```

### Resize Image

```python
resized = cv2.resize(img, (400, 400))
```

### Crop Image

```python
crop = img[100:400, 200:500]   # img[y1:y2, x1:x2]
```

### Drawing Shapes & Text

```python
# Rectangle
cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)

# Circle
cv2.circle(img, (300, 200), 50, (255, 0, 0), 3)

# Line
cv2.line(img, (0, 0), (400, 400), (0, 0, 255), 2)

# Text
cv2.putText(img, "OpenCV Workshop", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
```

---

## 📷 Webcam Video Capture

> Video = many images displayed rapidly (frames per second)

```python
import cv2

cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # Press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🌀 Image Filters & Edge Detection

### Gaussian Blur (Noise Removal)

```python
blur = cv2.GaussianBlur(frame, (15, 15), 0)
```

### Canny Edge Detection

Edges = boundaries of objects in the image.

```python
edges = cv2.Canny(frame, 100, 200)
cv2.imshow("Edges", edges)
```

**What it looks like:**

```
Original:         Edge Detection:
██████████        █          █
██████████   →    █          █
██████████        ████████████
```

---

## 🎨 Color Detection

RGB is sensitive to lighting — we use **HSV color space** instead for better color isolation.

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

### Detect Blue Objects (Full Example)

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 😊 Face Detection

OpenCV includes **pretrained Haar Cascade models** for face detection — no training needed!

```python
import cv2

face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**`detectMultiScale` Parameters:**

| Parameter | Meaning |
|---|---|
| `scaleFactor` | How much image is scaled down each pass |
| `minNeighbors` | Higher = fewer detections but more accurate |

---

## 🚀 YOLO Object Detection

**YOLO** = **You Only Look Once** — a deep learning model for real-time object detection.

**YOLO can detect:** People · Cars · Bottles · Phones · Chairs · Laptops · Animals · and 80+ more classes

**YOLO outputs for each object:**
- Object class (e.g. "person")
- Bounding box `(x, y, width, height)`
- Confidence score `(0.0 – 1.0)`

### Classical CV vs AI Vision

| Approach | Pipeline |
|---|---|
| Classical OpenCV | Image → Feature Extraction → Rule-based Detection |
| Deep Learning (YOLO) | Image → Neural Network → Object Detection |

| Method | How it works |
|---|---|
| Classical CV | Hand-crafted algorithms |
| Deep Learning | Neural network learns patterns from data |

### Run YOLO on an Image

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # Downloads weights automatically

results = model("image.jpg", show=True)
```

### YOLO Live Webcam Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) == 27:   # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🏋️ Workshop Activities

Try these in order — each builds on the last!

| # | Activity | Concepts Covered |
|---|---|---|
| 1 | Load and display an image | `imread`, `imshow` |
| 2 | Convert image to grayscale | `cvtColor` |
| 3 | Draw shapes and text | `rectangle`, `circle`, `putText` |
| 4 | Capture webcam video | `VideoCapture`, frame loop |
| 5 | Detect colored objects | HSV color space, masking |
| 6 | Detect faces | Haar Cascade classifier |
| 7 | Run YOLO object detection | Deep learning inference |

---

## 🎓 Learning Outcomes

By the end of this workshop, you will understand:

- [x] Basics of Computer Vision
- [x] How images are represented as numbers
- [x] Image processing operations with OpenCV
- [x] Webcam and video processing
- [x] Color detection using HSV
- [x] Face detection using Haar Cascades
- [x] AI-based object detection using YOLO

---

## 🎯 What Can You Build?

With OpenCV and YOLO, you can create:

- 📸 Smart cameras
- 🤖 Autonomous robots
- 🔐 AI surveillance systems
- 🕶️ Augmented reality apps
- 🏥 Medical image analysis tools

---
# 🎯 OpenCV Workshop — Project Collection

> **11 hands-on Computer Vision projects** built with Python, OpenCV, MediaPipe, and YOLO.  
> Each project is self-contained and progressively more advanced.

---

## 📋 Table of Contents

| # | Project | Difficulty | Libraries |
|---|---|---|---|
| 1 | [Edge Detection Camera](#1-edge-detection-camera) | 🟢 Beginner | OpenCV |
| 2 | [Blue Color Tracker](#2-blue-color-tracker) | 🟢 Beginner | OpenCV, NumPy |
| 3 | [Face Detection](#3-face-detection) | 🟢 Beginner | OpenCV |
| 4 | [Motion Detection](#4-motion-detection) | 🟡 Intermediate | OpenCV |
| 5 | [Live Blur Background (Portrait Mode)](#5-live-blur-background-portrait-mode) | 🟡 Intermediate | OpenCV |
| 6 | [Harry Potter Invisible Cloak](#6-harry-potter-invisible-cloak) | 🟡 Intermediate | OpenCV, NumPy |
| 7 | [YOLO Real-Time Detection](#7-yolo-real-time-webcam-detection) | 🟡 Intermediate | Ultralytics, OpenCV |
| 8 | [Object Counter using YOLO](#8-object-counter-using-yolo) | 🟡 Intermediate | Ultralytics, OpenCV |
| 9 | [Eye Controlled Mouse](#9-eye-controlled-mouse) | 🔴 Advanced | MediaPipe, PyAutoGUI |
| 10 | [Mouse Control using Eye Tracking](#10-mouse-control-using-eye-tracking) | 🔴 Advanced | MediaPipe, PyAutoGUI |
| 11 | [Virtual Keyboard with Hand Gestures](#11-virtual-keyboard-with-hand-gesture-control) | 🔴 Advanced | cvzone, pynput |

**Reference Sections:** [All Libraries](#-all-required-libraries) · [Learning Path](#️-suggested-learning-path) · [Troubleshooting](#-troubleshooting-guide)

---

---

## 1. Edge Detection Camera

> **Turns your webcam into a real-time sketch/edge-detection camera.**

**Difficulty:** 🟢 Beginner  
**Concepts:** Grayscale conversion, Canny edge detection, webcam loop

### How It Works

```
Webcam Frame → Grayscale → Canny Edge Detection → Display
```

Canny finds sudden changes in brightness — which are the **edges/outlines** of objects.

### Install

```bash
pip install opencv-python
```

### Code — `edge_detection.py`

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    cv2.imshow("Sketch Camera", edges)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Line | What it does |
|---|---|
| `cv2.VideoCapture(0)` | Opens default webcam |
| `cvtColor(...GRAY)` | Converts color frame to grayscale |
| `cv2.Canny(gray, 100, 200)` | Detects edges (threshold: 100 low, 200 high) |
| `cv2.imshow(...)` | Displays result |

### Try It Yourself
- Change `100, 200` to `50, 150` — notice more edges detected
- Change to `150, 250` — fewer, cleaner edges

**Press `Q` to quit.**

---

---

## 2. Blue Color Tracker

> **Detects and tracks any blue object in real-time using your webcam.**

**Difficulty:** 🟢 Beginner  
**Concepts:** HSV color space, masking, contour detection, bounding boxes

### How It Works

```
Webcam Frame → Convert to HSV → Create Blue Mask → Find Contours → Draw Box
```

We use **HSV** instead of RGB because HSV separates color (Hue) from brightness — making color detection more robust under different lighting.

### Install

```bash
pip install opencv-python numpy
```

### Code — `blue_color_tracker.py`

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Color Tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Step | Code | Purpose |
|---|---|---|
| 1 | `cvtColor(...HSV)` | Convert frame to HSV color space |
| 2 | `np.array([100,150,0])` | Define lower range of blue in HSV |
| 3 | `cv2.inRange(...)` | Create binary mask (white = blue pixels) |
| 4 | `cv2.findContours(...)` | Find connected regions in the mask |
| 5 | `area > 500` | Ignore tiny noise blobs |
| 6 | `cv2.rectangle(...)` | Draw box around detected object |

### Try It Yourself
- Hold a blue object in front of the webcam
- To track **red** objects, change the HSV range to `[0,120,70]` → `[10,255,255]`
- To track **green**, use `[40,50,50]` → `[80,255,255]`

**Press `Q` to quit.**

---

---

## 3. Face Detection

> **Detects human faces in real-time using OpenCV's built-in Haar Cascade model.**

**Difficulty:** 🟢 Beginner  
**Concepts:** Haar Cascade classifier, grayscale detection, bounding boxes

### How It Works

```
Webcam Frame → Grayscale → Haar Cascade Scan → Draw Boxes Around Faces
```

Haar Cascades are **pre-trained models** — OpenCV ships with them, no training needed!

### Install

```bash
pip install opencv-python
```

### Code — `face_detection.py`

```python
import cv2

face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Step | Code | Purpose |
|---|---|---|
| 1 | `CascadeClassifier(...)` | Load the pre-trained face model |
| 2 | `cvtColor(...GRAY)` | Convert to grayscale (required for Haar) |
| 3 | `detectMultiScale(gray, 1.3, 5)` | Scan image at multiple scales for faces |
| 4 | `for (x,y,w,h) in faces` | Loop over each detected face |
| 5 | `cv2.rectangle(...)` | Draw green box around each face |

**`detectMultiScale` Parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `scaleFactor` | `1.3` | Scale image by 30% each pass |
| `minNeighbors` | `5` | Minimum overlapping detections needed (higher = fewer false positives) |

### Try It Yourself
- Try `minNeighbors=2` — more detections but some false positives
- Try `minNeighbors=8` — fewer detections but more accurate
- Also try `haarcascade_eye.xml` to detect eyes!

**Press `Q` to quit.**

---

---

## 4. Motion Detection

> **Detects moving objects by comparing consecutive webcam frames.**

**Difficulty:** 🟡 Intermediate  
**Concepts:** Frame differencing, thresholding, blur

### How It Works

```
Frame 1 & Frame 2 → Absolute Difference → Blur → Threshold → White pixels = Motion
```

If a pixel changes significantly between two frames — **something moved!**

### Install

```bash
pip install opencv-python
```

### Code — `motion_detection.py`

```python
import cv2

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    cv2.imshow("Motion Detection", thresh)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Step | Code | Purpose |
|---|---|---|
| 1 | Read `frame1` and `frame2` | Capture two consecutive frames |
| 2 | `cv2.absdiff(frame1, frame2)` | Calculate pixel-by-pixel difference |
| 3 | `cvtColor(...GRAY)` | Convert diff to grayscale |
| 4 | `GaussianBlur(...)` | Smooth out small noise |
| 5 | `cv2.threshold(...20,255...)` | Any change > 20 → white pixel (motion!) |
| 6 | Shift frames | `frame1 = frame2`, read new `frame2` |

### Try It Yourself
- Sit still — the screen should be nearly black
- Wave your hand — white regions appear where motion is detected
- Change threshold from `20` to `50` — less sensitive

**Press `Q` to quit.**

---

---

## 5. Live Blur Background (Portrait Mode)

> **Blurs everything except your face — just like smartphone Portrait Mode!**

**Difficulty:** 🟡 Intermediate  
**Concepts:** Gaussian blur, face detection, region replacement (masking)

### How It Works

```
Webcam Frame → Blur Entire Frame → Detect Face → Replace Face Region with Original (Sharp)
```

### Install

```bash
pip install opencv-python
```

### Code — `Live_Blur_Background.py`

```python
import cv2

cap = cv2.VideoCapture(0)

face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (51, 51), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.3, 5)

    mask = blur.copy()

    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]

    cv2.imshow("Portrait Mode", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Step | Code | Purpose |
|---|---|---|
| 1 | `GaussianBlur(frame, (51,51), 0)` | Create fully blurred version |
| 2 | `detectMultiScale(...)` | Detect face location |
| 3 | `mask = blur.copy()` | Start with blurred frame |
| 4 | `mask[y:y+h, x:x+w] = frame[...]` | Paste the sharp face region back in |

### Try It Yourself
- Increase blur to `(101,101)` for stronger background blur
- Decrease to `(21,21)` for subtle blur
- Move around — the face region stays sharp!

**Press `Q` to quit.**

---

---

## 6. Harry Potter Invisible Cloak

> **Make a red cloth "invisible" — replaces it with the background in real-time!**

**Difficulty:** 🟡 Intermediate  
**Concepts:** Background capture, HSV masking, bitwise operations, morphology

### How It Works

```
Capture Background First → Detect Red Color → Replace Red Region with Background
```

> ⚠️ **Important:** Stay out of frame for the first 2 seconds while the background is captured!

### Install

```bash
pip install opencv-python numpy
```

### Code — `Harry_Potter_Invisible_Cloak.py`

```python
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

time.sleep(2)  # Wait 2 seconds — stay out of frame!

background = None
for i in range(30):
    ret, background = cap.read()

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    mask_inv = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask)       # Background where red is

    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)             # Original frame where no red

    final = cv2.add(res1, res2)                                     # Combine both

    cv2.imshow("Invisible Cloak", final)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Step | What Happens |
|---|---|
| Background capture | 30 frames captured while no one is in frame |
| HSV masking | Red pixels are isolated into a white mask |
| Morphology cleanup | Small noise in the mask is removed |
| `mask_inv` | Inverted mask — everything that is NOT red |
| `res1` | Background image shown where red cloth is |
| `res2` | Live frame shown everywhere else |
| `cv2.add(res1, res2)` | Combine both — cloak effect! |

### Tips for Best Results
- Use a **bright red** cloth or dupatta
- Ensure **good, even lighting**
- Keep the **background plain** (wall, etc.)
- Stay out of frame when the program starts!

**Press `Q` to quit.**

---

---

## 7. YOLO Real-Time Webcam Detection

> **Detect 80+ types of objects in real-time using the YOLOv8 AI model.**

**Difficulty:** 🟡 Intermediate  
**Concepts:** Deep learning inference, YOLO model, annotated frame output

### How It Works

```
Webcam Frame → YOLOv8 Neural Network → Bounding Boxes + Labels + Confidence → Display
```

YOLO processes the entire image **once** (You Only Look Once) — making it extremely fast.

### Install

```bash
pip install ultralytics opencv-python
```

> 📥 The `yolov8n.pt` model (~6MB) downloads automatically on first run.

### Code — `YOLO_Real-Time_Webcam_Detection.py`

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) == 27:   # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Line | Purpose |
|---|---|
| `YOLO("yolov8n.pt")` | Load YOLOv8 nano model (fastest, smallest) |
| `model(frame)` | Run detection on current frame |
| `results[0].plot()` | Draw bounding boxes and labels automatically |

### YOLO Model Sizes

| Model | Speed | Accuracy | Use When |
|---|---|---|---|
| `yolov8n.pt` | ⚡ Fastest | Good | Workshop / low-end PC |
| `yolov8s.pt` | Fast | Better | Everyday use |
| `yolov8m.pt` | Moderate | High | Production |

**Press `ESC` to quit.**

---

---

## 8. Object Counter using YOLO

> **Counts the total number of objects detected in the frame in real-time.**

**Difficulty:** 🟡 Intermediate  
**Concepts:** YOLO detection, counting bounding boxes, text overlay

### How It Works

```
YOLO Detection → Count Number of Boxes → Display Count on Frame
```

### Install

```bash
pip install ultralytics opencv-python
```

### Code — `Object_Counter_using_YOLO.py`

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    count = len(results[0].boxes)

    cv2.putText(annotated_frame,
                f"Objects: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Object Counter", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Line | Purpose |
|---|---|
| `results[0].boxes` | List of all detected bounding boxes |
| `len(results[0].boxes)` | Count of all detected objects |
| `cv2.putText(...)` | Display count on the frame |

### Try It Yourself
- Place multiple objects (bottle, phone, book) in frame
- Count updates every frame in real-time
- Try counting only specific classes (e.g. only persons):

```python
# Count only persons (class id = 0)
person_count = sum(1 for box in results[0].boxes if int(box.cls) == 0)
```

**Press `ESC` to quit.**

---

---

## 9. Eye Controlled Mouse

> **Control your mouse cursor by moving your eyes, and click by blinking!**

**Difficulty:** 🔴 Advanced  
**Concepts:** MediaPipe Face Mesh, landmark detection, eye tracking, PyAutoGUI

### How It Works

```
Webcam → MediaPipe Face Mesh → Track Eye Landmarks → Map to Screen → Blink = Click
```

MediaPipe detects **468 facial landmarks** — we use specific ones around the eye to track position and detect blinks.

### Install

```bash
pip install opencv-python mediapipe pyautogui
```

> ⚠️ **Note:** PyAutoGUI controls your actual mouse. The window may move around!

### Code — `eye_controlled_mouse.py`

```python
import cv2
import mediapipe as mp
import pyautogui
import math

def euclidean_distance(landmark1, landmark2, frame_w, frame_h):
    x1, y1 = int(landmark1.x * frame_w), int(landmark1.y * frame_h)
    x2, y2 = int(landmark2.x * frame_w), int(landmark2.y * frame_h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                screen_x = int(screen_w * landmark.x)
                screen_y = int(screen_h * landmark.y)
                pyautogui.moveTo(screen_x, screen_y)

        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        eye_open_distance = euclidean_distance(left_eye_top, left_eye_bottom, frame_w, frame_h)

        if eye_open_distance < 5:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

### Step-by-Step Explanation

| Concept | Landmarks Used | Purpose |
|---|---|---|
| Eye position | `landmarks[474:478]` | Track right iris position → move mouse |
| Blink detection | `landmarks[159]` (top) & `landmarks[145]` (bottom) | Measure eye opening distance |
| Click trigger | `eye_open_distance < 5` | Small distance = eye closed = click! |

### Tips
- Sit in **good lighting** facing the camera
- Keep your **head still** — move only your eyes
- Blink **slowly and deliberately** to click
- Adjust the blink threshold `< 5` if click sensitivity is off

**Press `Q` to quit.**

---

---

## 10. Mouse Control using Eye Tracking

> **An enhanced version of Eye Controlled Mouse** — same concept with cleaner structure.

**Difficulty:** 🔴 Advanced

> This project uses the exact same code and concepts as **Project 9**. Refer to the steps above.

The key difference is how the project is organized for clarity — making it easier to modify and extend (e.g. adding right-eye tracking, multiple gestures, etc.)

---

---

## 11. Virtual Keyboard with Hand Gesture Control

> **A full virtual keyboard rendered on screen — type by pointing at keys and pinching your fingers!**

**Difficulty:** 🔴 Advanced  
**Concepts:** Hand landmark detection, gesture recognition, virtual UI, keyboard simulation

### How It Works

```
Webcam → cvzone Hand Detector → Finger Position → Hover Key → Pinch Gesture = Key Press
```

### Install

```bash
pip install opencv-python cvzone mediapipe pynput
```

> 💡 cvzone is a wrapper around MediaPipe that makes hand tracking much simpler.

### Code — `Virtual-Keyboard-with-Hand-Gesture-Control.py`

> *(See uploaded file — code is ~170 lines)*

### Step-by-Step Explanation

#### 1. Setup

```python
cap.set(3, 1280)   # Set webcam width
cap.set(4, 720)    # Set webcam height
detector = HandDetector(detectionCon=0.8, maxHands=2)
```

#### 2. Keyboard Layout

```python
keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"],
        ["SAVE", " ", "CLEAR"]]
```

#### 3. Key Hover Detection

```python
if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y + h:
    # Finger tip (landmark 8) is over this button → highlight it
```

#### 4. Pinch = Key Press

```python
if check_pinch(lmList1):
    keyboard.press(button.text)
    finalText += button.text
```

#### 5. Gesture Controls

| Gesture | Action |
|---|---|
| Point at key | Highlights the key |
| Pinch (any 2 fingertips close) | Types the key |
| Point at SAVE + Pinch | Saves typed text to a `.txt` file |
| Point at CLEAR + Pinch | Clears the text |
| Make a Fist ✊ | Saves & closes the program |

### Tips
- Keep your hand **clearly visible** to the camera
- Good, **even lighting** improves detection
- Try typing your name!

**Press `ESC` or make a fist to exit.**

---

---

## 📦 All Required Libraries

```bash
pip install opencv-python numpy ultralytics mediapipe pyautogui cvzone pynput
```

| Library | Used In |
|---|---|
| `opencv-python` | All projects |
| `numpy` | Color Tracker, Invisible Cloak |
| `ultralytics` | YOLO Detection, Object Counter |
| `mediapipe` | Eye Mouse, Virtual Keyboard |
| `pyautogui` | Eye Mouse |
| `cvzone` | Virtual Keyboard |
| `pynput` | Virtual Keyboard |

---

## 🗺️ Suggested Learning Path

```
Project 1 → 2 → 3      (Beginner: basic OpenCV)
      ↓
Project 4 → 5 → 6      (Intermediate: frame ops + masking)
      ↓
Project 7 → 8           (AI: YOLO detection)
      ↓
Project 9 → 10 → 11    (Advanced: MediaPipe + gesture control)
```

---

## ❓ Troubleshooting Guide

---

### 📷 Webcam & Camera Issues

| Problem | Fix |
|---|---|
| Webcam not opening | Change `VideoCapture(0)` to `VideoCapture(1)` or `(2)` — laptop may have multiple cameras |
| Webcam already in use | Close Zoom, Teams, or any other app using the camera, then re-run |
| First frame is black | Add `time.sleep(1)` before your `while` loop to let the camera warm up |
| Webcam LED stays on after closing | Always include `cap.release()` and `cv2.destroyAllWindows()` at the end |

```python
# Always end your program with these two lines
cap.release()
cv2.destroyAllWindows()
```

---

### 📦 Installation Issues

| Problem | Fix |
|---|---|
| `pip` installs to wrong Python | Use `pip3` instead of `pip` |
| OpenCV import error | Run `pip uninstall opencv-python opencv-contrib-python` then reinstall one only |
| MediaPipe not installing | MediaPipe supports Python **3.9 – 3.11** only — Python 3.12+ may fail |
| cvzone errors | Install specifically `pip install cvzone==1.5.6` |
| `ModuleNotFoundError` | Double-check you installed in the same environment PyCharm is using |

> 💡 **Tip:** In PyCharm, open the terminal at the bottom and run `pip install` from there — it installs into the correct project environment automatically.

---

### 🤖 YOLO Issues

| Problem | Fix |
|---|---|
| First run hangs / freezes | It's downloading the model (~6MB) — just wait, it only downloads once |
| Very slow / low FPS | Switch to `yolov8n.pt` (nano) — it's the fastest model |
| CUDA / GPU errors | Add `device='cpu'` to force CPU mode (see below) |
| Detection seems inaccurate | Try `yolov8s.pt` (small) for better accuracy at slight speed cost |

```python
# Force CPU mode if GPU errors appear
results = model(frame, device='cpu')
```

---

### 👁️ MediaPipe & Eye Tracking Issues

| Problem | Fix |
|---|---|
| Mouse flies to corner and gets stuck | PyAutoGUI failsafe is triggering — add the line below at the top of your file |
| Blink not registering | Increase threshold — change `< 5` to `< 8` and test |
| Clicking too much / too sensitive | Decrease threshold — change `< 5` to `< 3` |
| Face mesh not detecting | Ensure you convert to RGB **before** `face_mesh.process()` — BGR will not work |

```python
# Fix mouse getting stuck in corner
import pyautogui
pyautogui.FAILSAFE = False
```

```python
# CORRECT — MediaPipe needs RGB, not BGR
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
output = face_mesh.process(rgb_frame)
```

---

### 🧥 Invisible Cloak Issues

| Problem | Fix |
|---|---|
| Cloak effect not working | Make sure you stay **out of frame** during the first 2 seconds of startup |
| Background captured incorrectly | Press `Q`, rerun, and immediately step aside |
| Red detection picking up skin | Tighten the HSV range — reduce `upper_red` saturation |
| Flickering / noisy edges | Increase morphology kernel from `(3,3)` to `(5,5)` |

```python
# Stronger noise removal for smoother cloak edges
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
```

> 💡 **Best results:** Bright red cloth + plain background + even lighting

---

### ⌨️ Virtual Keyboard Issues

| Problem | Fix |
|---|---|
| Keys registering multiple times | Increase `CLICK_DELAY = 0.3` to `CLICK_DELAY = 0.5` |
| Hand not detected | Ensure hand is fully visible and well-lit, not cut off at frame edges |
| `pynput` blocked on macOS | Go to System Preferences → Security & Privacy → Accessibility → allow Terminal / PyCharm |
| Program very laggy | Lower webcam resolution: change `cap.set(3, 1280)` to `cap.set(3, 640)` |

---

### 🖥️ Operating System Specific Issues

| OS | Problem | Fix |
|---|---|---|
| **Windows** | `cv2.imshow()` window freezes | Make sure `cv2.waitKey(1)` is inside the loop |
| **macOS** | Camera permission denied | System Preferences → Security → Camera → allow your IDE |
| **macOS** | PyAutoGUI blocked | System Preferences → Security → Accessibility → allow Terminal |
| **Linux** | `cv2.imshow()` crashes | Run with `DISPLAY=:0` or switch to an X11 session (not Wayland) |
| **Linux** | PyAutoGUI doesn't work | Switch from Wayland to X11 at the login screen |

---

### 🐛 Common Coding Mistakes

| Mistake | What Happens | Fix |
|---|---|---|
| `cv2.waitKey(0)` inside video loop | Program freezes on first frame | Use `cv2.waitKey(1)` instead |
| Forgetting `cap.release()` | Webcam stays on, next run fails | Always add it at the end |
| Wrong axis order in `cv2.resize()` | Stretched/squished image | It's `(width, height)` — opposite of NumPy |
| Confused by `img.shape` output | Wrong assumptions about dimensions | Shape returns `(height, width, channels)` |
| BGR vs RGB mix-up | Wrong colors / MediaPipe fails | OpenCV uses BGR; MediaPipe needs RGB |

```python
# ❌ WRONG — freezes on first frame in a video loop
cv2.waitKey(0)

# ✅ CORRECT — waits 1ms then continues loop
cv2.waitKey(1)

# ❌ WRONG axis order
resized = cv2.resize(img, (height, width))

# ✅ CORRECT axis order
resized = cv2.resize(img, (width, height))

# Remember: img.shape → (height, width, channels)
h, w, c = img.shape
```

---

### 🆘 Still Stuck?

1. Read the **full error message** in the terminal — it usually tells you exactly what's wrong
2. Google the **exact error text** — most OpenCV errors are well documented
3. Check that your **Python version** is 3.9–3.11
4. Check that you're running the file from **inside PyCharm** (not double-clicking the `.py` file)
5. Try **restarting PyCharm** — sometimes the kernel holds old state


---

<div align="center">

**Happy Coding! 🚀**

*OpenCV Computer Vision Workshop*

</div>
