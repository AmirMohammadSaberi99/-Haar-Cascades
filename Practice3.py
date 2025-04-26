import cv2
import matplotlib.pyplot as plt

def detect_smiles_on_faces(
    img,
    face_xml=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    smile_xml=cv2.data.haarcascades + 'haarcascade_smile.xml',
    face_params=None,
    smile_params=None
):
    # default detection params
    face_params  = face_params or {'scaleFactor':1.1, 'minNeighbors':5,  'minSize':(30,30)}
    smile_params = smile_params or {'scaleFactor':1.3, 'minNeighbors':10, 'minSize':(15,15)}

    face_cascade  = cv2.CascadeClassifier(face_xml)
    smile_cascade = cv2.CascadeClassifier(smile_xml)
    if face_cascade.empty() or smile_cascade.empty():
        raise IOError("Could not load one of the cascades")

    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) detect faces
    faces = face_cascade.detectMultiScale(gray, **face_params)
    total_smiles = 0

    for (x, y, w, h) in faces:
        # 2) focus smile detection on the lower half of the face
        y0 = y + h//2
        roi_gray  = gray[y0:y+h, x:x+w]
        roi_color = out [y0:y+h, x:x+w]

        # 3) equalize contrast in ROI
        roi_gray = cv2.equalizeHist(roi_gray)

        # 4) detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, **smile_params)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx+sw, sy+sh),
                (0, 0, 255), 2
            )
        total_smiles += len(smiles)

    return out, total_smiles

if __name__ == '__main__':
    img = cv2.imread('Test.jpg')
    annotated, count = detect_smiles_on_faces(img)

    print(f"Smiles detected: {count}")
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(f"Smiles detected: {count}")
    plt.show()
