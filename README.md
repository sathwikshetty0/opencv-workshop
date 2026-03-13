
## What is OpenCV

OpenCV stands for **Open Source Computer Vision Library**.

It provides **thousands of functions** for:

* Image processing
* Object detection
* Face detection
* Motion tracking
* Camera calibration
* Robotics vision

---

## Why Use Python for OpenCV

Python is ideal because:

* Easy syntax
* Large ecosystem
* Quick prototyping
* Strong AI/ML support

---

# 2️⃣ Environment Setup 

## Install Python

Download from:

[https://www.python.org](https://www.python.org)

Verify installation:

```bash
python --version
```

---

## Install OpenCV

Install using pip:

```bash
pip install opencv-python
```

Optional packages:

```bash
pip install numpy matplotlib
```

---

## Test Installation

```python
import cv2

print(cv2.__version__)
```

If a version number appears, installation is successful.

---

# 3️⃣ First OpenCV Program (Image Display)

## Code

```python
import cv2

# Load image
img = cv2.imread("sample.jpg")

# Display image
cv2.imshow("Image Window", img)

# Wait until key pressed
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()
```

---

## Explanation

| Function                  | Purpose             |
| ------------------------- | ------------------- |
| `cv2.imread()`            | Reads image file    |
| `cv2.imshow()`            | Displays image      |
| `cv2.waitKey()`           | Waits for key press |
| `cv2.destroyAllWindows()` | Closes windows      |

---

## Image Representation

Images are stored as **arrays of pixels**.

Example:

```python
print(img.shape)
```

Output example:

```
(720, 1280, 3)
```

Meaning:

| Value | Meaning        |
| ----- | -------------- |
| 720   | Height         |
| 1280  | Width          |
| 3     | Color channels |

---

# 4️⃣ Basic Image Processing (40 min)

## Convert to Grayscale

Grayscale images are easier for processing.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray)
```

---

## Resize Image

```python
resized = cv2.resize(img, (400,400))
```

---

## Crop Image

```python
crop = img[100:400, 200:500]
```

---

# 5️⃣ Drawing on Images

## Draw Rectangle

```python
cv2.rectangle(img,(50,50),(200,200),(0,255,0),3)
```

---

## Draw Circle

```python
cv2.circle(img,(300,200),50,(255,0,0),3)
```

---

## Draw Line

```python
cv2.line(img,(0,0),(400,400),(0,0,255),2)
```

---

## Add Text

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

## Student Activity

Ask students to:

* Load an image
* Draw rectangle
* Write their name on image

---

# 6️⃣ Working with Webcam 

Real-time video processing.

## Code

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

---

## Explanation

| Function          | Purpose              |
| ----------------- | -------------------- |
| `VideoCapture(0)` | Opens default webcam |
| `read()`          | Captures frame       |
| `imshow()`        | Shows video          |

---

# 7️⃣ Image Filters and Edge Detection 

## Gaussian Blur

Used to remove noise.

```python
blur = cv2.GaussianBlur(frame,(15,15),0)
```

---

## Edge Detection

Edges represent object boundaries.

```python
edges = cv2.Canny(frame,100,200)
```

---

## Show Both

```python
cv2.imshow("Original", frame)
cv2.imshow("Edges", edges)
```

---

# 8️⃣ Color Detection 

## Why Use HSV Color Space

RGB is sensitive to lighting.

HSV separates **color from brightness**.

---

## Convert to HSV

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

---

## Detect Blue Color

```python
lower_blue = (100,150,0)
upper_blue = (140,255,255)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
```

---

## Extract Blue Objects

```python
result = cv2.bitwise_and(frame, frame, mask=mask)
```

---

## Full Code

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

# 9️⃣ Face Detection 

OpenCV provides pretrained models called **Haar Cascades**.

---

## Code

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

        cv2.rectangle(frame,
                      (x,y),
                      (x+w,y+h),
                      (0,255,0),
                      2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Explanation

| Parameter    | Meaning            |
| ------------ | ------------------ |
| scaleFactor  | image scaling      |
| minNeighbors | detection accuracy |
| rectangle    | draws face box     |

---

---
