


# OpenCV Computer Vision Workshop

## Introduction

This workshop introduces students to **Computer Vision using Python**.  
Participants will learn how computers process images, detect objects, and analyze video streams.
---
# 1️⃣ What is Computer Vision

Computer Vision is a field of Artificial Intelligence that enables computers to **interpret and understand visual information from the world**.

Examples:

- Face unlock in smartphones
- Self-driving cars
- Security surveillance
- Medical image analysis
- Augmented reality filters

In simple words:

> Computer Vision = Teaching computers how to **see**.

# 2️⃣ What is OpenCV

OpenCV stands for **Open Source Computer Vision Library**.

It provides **thousands of functions** for:

- Image processing
- Object detection
- Face detection
- Motion tracking
- Camera calibration
- Robotics vision

OpenCV is widely used in:

- Robotics
- Artificial Intelligence
- Autonomous vehicles
- Medical imaging
- Industrial automation


---

```markdown
# 🧠 How OpenCV Works Internally (Image Processing Pipeline)

Before writing code, it is useful to understand **what actually happens inside OpenCV** when we process an image.

OpenCV processes images using a **pipeline of operations**.

Think of it like a **factory assembly line** where each stage modifies the image.

---

## Image Processing Pipeline

```


````

Each stage performs a different task.

---

#  Image Acquisition

This is the step where OpenCV **gets the image**.

The image may come from:

- Camera
- Image file
- Video stream
- Dataset

Example:

```python
img = cv2.imread("image.jpg")
````

or

```python
cap = cv2.VideoCapture(0)
```

Fun explanation:

> This is like **opening your eyes**.

---

# Image Representation

Once the image is loaded, OpenCV stores it as a **matrix of numbers**.

Example grayscale image:

```

0   50  100
120 200 180
30  60  90

```

Each number represents **brightness of a pixel**.

Range:

```

0 → black  
255 → white

```

For color images OpenCV stores:

```

[Blue, Green, Red]

```

Example pixel:

```

[255,0,0] → Blue  
[0,255,0] → Green  
[0,0,255] → Red

```

Image shape:

```

(height, width, channels)

```

Example:

```

(720,1280,3)

```

---

# Preprocessing

Before detecting objects, images often need **cleaning or transformation**.

Typical preprocessing steps:

* Resize image
* Convert color spaces
* Remove noise
* Blur image

Example:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

or

```python
blur = cv2.GaussianBlur(img,(15,15),0)
```

Fun analogy:

> Preprocessing is like **cleaning your glasses before looking at something carefully**.

---

# Feature Detection

Now OpenCV tries to **identify useful patterns in the image**.

Examples of features:

* Edges
* Corners
* Shapes
* Keypoints
* Color regions

Example:

```python
edges = cv2.Canny(img,100,200)
```

Edge detection finds **object boundaries**.

Simple visualization:

```

Original Image

██████████

Edge Detection

█      █
█      █
████████

```

---

#  Object Detection / Recognition

Once features are extracted, OpenCV can **detect objects**.

Traditional methods include:

* Haar Cascades
* HOG detectors
* Template matching

Example: Face detection

```python
faces = face.detectMultiScale(gray,1.3,5)
```

This returns coordinates of faces:

```

(x, y, width, height)

```

OpenCV then draws rectangles around them.

---

#  Visualization

Finally the processed result is displayed.

Example:

```python
cv2.imshow("Result", frame)
```

This step allows us to **see the output of the pipeline**.

---

# Complete Example Pipeline

When you run a face detection program, internally this happens:

```

Camera Frame
│
▼
Convert to Grayscale
│
▼
Apply Haar Cascade
│
▼
Detect Faces
│
▼
Draw Bounding Boxes
│
▼
Display Result

```

---

# Example Code Pipeline

```python
import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,100,200)

    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

# Classical Computer Vision vs AI Vision

Traditional OpenCV methods:

```

Image → Feature Extraction → Rule-based Detection

```

Modern AI methods (YOLO):

```

Image → Neural Network → Object Detection

```

Difference:

| Method        | Approach                       |
| ------------- | ------------------------------ |
| Classical CV  | Hand-crafted algorithms        |
| Deep Learning | Neural networks learn patterns |

---

# Fun Way to Explain to Students

Human Vision:

```

Eye → Brain → Recognize Object

```

Computer Vision:

```

Camera → OpenCV → AI Model → Object Recognition

```

Fun line to tell students:

> Humans see using **eyes and brain**
> Computers see using **cameras and math**.

---


# 3️⃣ Why Use Python for OpenCV

Python is ideal because:

- Simple and readable syntax
- Large ecosystem
- Rapid prototyping
- Strong AI and Machine Learning libraries

---

# 4️⃣ Environment Setup

## Install Python

Download from:

https://www.python.org

Verify installation:

```bash
python --version
````

---

# 5️⃣ Install PyCharm (IDE)

Download:

[https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)

Install **PyCharm Community Edition**.

Steps:

1. Open PyCharm
2. Click **New Project**
3. Select **Pure Python**
4. Create project

---

# 6️⃣ Install Required Libraries

Open terminal in PyCharm and run:

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install ultralytics
```

Or install everything together:

```bash
pip install opencv-python numpy matplotlib ultralytics
```

---

# 7️⃣ Verify Installation

Create file `test_setup.py`

```python
import cv2
import numpy as np
from ultralytics import YOLO

print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)
print("YOLO Library Loaded Successfully")
```

If no errors appear, the environment is ready.

---

# 8️⃣ Project Folder Structure

```
opencv-workshop/

images/
sample.jpg

scripts/
image_display.py
webcam.py
color_detection.py
face_detection.py

models/

requirements.txt
```

---

# 9️⃣ Understanding Digital Images

Images are stored as **arrays of pixels**.

Example grayscale matrix:

```
120 100 90
200 210 180
30  40  50
```

Pixel value range:

```
0 → black
255 → white
```

---

## Color Images

Color images have **3 channels**:

```
BGR (Blue, Green, Red)
```

Example pixel:

```
[255,0,0] → Blue
[0,255,0] → Green
[0,0,255] → Red
```

Image shape example:

```python
print(img.shape)
```

Output:

```
(720, 1280, 3)
```

Meaning:

| Value | Meaning        |
| ----- | -------------- |
| 720   | Height         |
| 1280  | Width          |
| 3     | Color Channels |

---

# 🔟 First OpenCV Program

```python
import cv2

img = cv2.imread("sample.jpg")

cv2.imshow("Image Window", img)

cv2.waitKey(0)

cv2.destroyAllWindows()
```

Explanation:

| Function            | Purpose             |
| ------------------- | ------------------- |
| imread()            | Reads image         |
| imshow()            | Displays image      |
| waitKey()           | Waits for key press |
| destroyAllWindows() | Closes windows      |

---

# 1️⃣1️⃣ Basic Image Processing

## Convert to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray)
```

---

## Resize Image

```python
resized = cv2.resize(img,(400,400))
```

---

## Crop Image

```python
crop = img[100:400,200:500]
```

---

# 1️⃣2️⃣ Drawing on Images

## Rectangle

```python
cv2.rectangle(img,(50,50),(200,200),(0,255,0),3)
```

## Circle

```python
cv2.circle(img,(300,200),50,(255,0,0),3)
```

## Line

```python
cv2.line(img,(0,0),(400,400),(0,0,255),2)
```

## Text

```python
cv2.putText(img,
            "OpenCV Workshop",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2)
```

---

# 1️⃣3️⃣ Webcam Video Capture

```python
import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Video is simply **many images displayed rapidly**.

---

# 1️⃣4️⃣ Image Filters

## Gaussian Blur

Removes noise.

```python
blur = cv2.GaussianBlur(frame,(15,15),0)
```

---

# 1️⃣5️⃣ Edge Detection

Edges represent boundaries of objects.

```python
edges = cv2.Canny(frame,100,200)

cv2.imshow("Edges", edges)
```

---

# 1️⃣6️⃣ Color Detection

RGB is sensitive to lighting.

Instead we use **HSV color space**.

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

---

## Blue Object Detection

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])

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

# 1️⃣7️⃣ Face Detection

OpenCV provides **pretrained Haar Cascade models**.

```python
import cv2

face = cv2.CascadeClassifier(
cv2.data.haarcascades +
"haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Parameters:

| Parameter    | Meaning            |
| ------------ | ------------------ |
| scaleFactor  | image scaling      |
| minNeighbors | detection accuracy |

---

# 1️⃣8️⃣ Introduction to YOLO

YOLO stands for:

**You Only Look Once**

It is a **deep learning model used for object detection**.

YOLO can detect:

* People
* Cars
* Bottles
* Phones
* Chairs
* Laptops
* Animals

YOLO predicts:

* Object class
* Bounding box
* Confidence score

---

# 1️⃣9️⃣ YOLO Object Detection Example

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("image.jpg", show=True)
```

The model automatically downloads weights.

---

# 2️⃣0️⃣ YOLO Webcam Detection

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

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Press **ESC** to exit.

---

# 2️⃣1️⃣ Workshop Activities

Students should try:

### Activity 1

Load and display an image.

### Activity 2

Convert image to grayscale.

### Activity 3

Draw shapes and text.

### Activity 4

Capture webcam video.

### Activity 5

Detect colored objects.

### Activity 6

Detect faces.

### Activity 7

Run YOLO object detection.

---

# 2️⃣2️⃣ Learning Outcomes

By the end of the workshop students will understand:

* Basics of Computer Vision
* Image representation
* Image processing using OpenCV
* Webcam video processing
* Color detection techniques
* Face detection
* AI based object detection using YOLO

---

# 🎯 Conclusion

Computer Vision allows machines to interpret visual data from the world.

Using OpenCV and modern AI models like YOLO, developers can build applications such as:

* Smart cameras
* Autonomous robots
* AI surveillance
* Augmented reality systems
* Medical image analysis

```

