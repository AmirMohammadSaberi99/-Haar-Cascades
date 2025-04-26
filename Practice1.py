import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_faces_haar(
    img,
    cascade_paths=None,
    scaleFactor: float = 1.05,
    minNeighbors: int = 4,
    minSize: tuple = (30, 30),
    clahe_clip: float = 2.0,
    clahe_grid: tuple = (8, 8),
    groupThreshold: int = 2,
    eps: float = 0.2
):
    """
    Detect faces using one or more Haar cascades with CLAHE + grouping.

    Returns:
      faces: list of (x, y, w, h) after non-max suppression
      img_annotated: BGR image with boxes drawn
    """
    # 1) Preprocess: grayscale + CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_eq = clahe.apply(gray)

    # 2) Load cascades
    if cascade_paths is None:
        base = cv2.data.haarcascades
        cascade_paths = [
            base + 'haarcascade_frontalface_default.xml',
            base + 'haarcascade_frontalface_alt2.xml',
            base + 'haarcascade_frontalface_alt_tree.xml'
        ]

    rects = []
    for path in cascade_paths:
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise IOError(f"Could not load Haar cascade: {path}")
        hits = cascade.detectMultiScale(
            gray_eq,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        rects.extend(hits)

    # 3) Group/merge overlapping rects
    #    groupThreshold = how many overlapping rects at minimum
    #    eps = how similar the rectangles must be
    if len(rects) > 0:
        grouped, _ = cv2.groupRectangles(rects, groupThreshold, eps)
    else:
        grouped = []

    # 4) Draw on a copy
    out = img.copy()
    for (x, y, w, h) in grouped:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return grouped, out

if __name__ == '__main__':
    # load the image (painting with many faces)
    img = cv2.imread('Test.jpg')
    if img is None:
        raise FileNotFoundError("Image not found, check the path!")

    # run detection
    faces, annotated = detect_faces_haar(
        img,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
        clahe_clip=3.0,
        clahe_grid=(8,8),
        groupThreshold=2,
        eps=0.2
    )

    print(f"Faces detected: {len(faces)}")

    # display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(annotated_rgb)
    plt.axis('off')
    plt.title(f"Faces detected: {len(faces)}")
    plt.show()
