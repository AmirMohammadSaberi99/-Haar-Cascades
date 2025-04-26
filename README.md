# Face & Smile Detection with Haar Cascades

This repository contains three Python scripts that demonstrate classical computer vision techniques for detecting faces and smiles using OpenCV's Haar cascades.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [File Descriptions](#file-descriptions)
- [Usage Examples](#usage-examples)
- [Parameters & Tuning](#parameters--tuning)
- [License](#license)

---

## Overview

1. **Practice1.py**  
   A robust batch detector that uses multiple Haar cascades, CLAHE preprocessing, and non-max suppression to find faces in a static image.

2. **Practice2.py**  
   A real-time webcam application that runs a Haar cascade on each frame and draws green bounding boxes around detected faces.

3. **Practice3.py**  
   A two-stage detector that first finds faces in an image, then applies a smile cascade (on the lower half of each face) with contrast equalization to highlight smiles only.

---

## Prerequisites

- Python 3.7+  
- OpenCV with Python bindings:  
  ```bash
  pip install opencv-python
  ```
- For plotting (static scripts only):  
  ```bash
  pip install matplotlib
  ```

---

## File Descriptions

### `detect_faces_haar.py`

- **Function**: `detect_faces_haar(img, ...)`  
  - Applies CLAHE (contrast-limited adaptive histogram equalization).  
  - Runs three different frontal-face Haar cascades.  
  - Uses `cv2.groupRectangles` to merge overlapping detections.  
  - Returns a list of face rectangles and an annotated image.

- **Standalone**: When run as `__main__`, reads `Test.jpg`, detects faces, prints the count, and displays the result with Matplotlib.


### `realtime_face_detection.py`

- **Function**: `main()`  
  - Opens the default webcam (`cv2.VideoCapture(0)`).  
  - Loads a single `haarcascade_frontalface_default.xml`.  
  - For each frame, converts to grayscale, runs `detectMultiScale`, and draws green rectangles.
  - Displays the video in a window until **q** is pressed.


### `detect_smiles_on_faces.py`

- **Function**: `detect_smiles_on_faces(img, ...)`  
  - Loads face and smile cascades.  
  - Detects faces, then focuses on the lower half of each face.  
  - Applies histogram equalization to the mouth region.  
  - Runs a smile detector and draws red rectangles for smiles only.

- **Standalone**: When run as `__main__`, reads `Test.jpg`, prints the smile count, and displays the annotated image.

---

## Usage Examples

### Static Face Detection
```bash
python detect_faces_haar.py --image path/to/your_image.jpg
```

### Real-Time Webcam Detection
```bash
python realtime_face_detection.py
```

### Smile Detection on Faces
```bash
python detect_smiles_on_faces.py --image path/to/your_image.jpg
```

*(Note: adjust the script names or add `argparse` as needed for custom paths.)*

---

## Parameters & Tuning

Each script exposes key parameters you can tweak directly in code or via function arguments:

- **scaleFactor**: How much the image size is reduced at each scale. Lower → finer search.
- **minNeighbors**: How many neighbors a rectangle needs to retain it. Lower → more detections (and more false positives).
- **minSize**: Minimum object size (width, height) to detect.
- **CLAHE** (`clipLimit`, `tileGridSize`): Controls local contrast amplification.
- **groupRectangles** (`groupThreshold`, `eps`): Merges overlapping boxes to reduce duplicates.

Adjust these to balance sensitivity vs. robustness for your specific images or video feed.

---

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.

