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

---

<div align="center">

**Happy Coding! 🚀**

*OpenCV Computer Vision Workshop*

</div>
