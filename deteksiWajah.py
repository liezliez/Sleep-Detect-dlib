import cv2

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

def deteksiWajah(frame):

    frame.flags.writeable = True

# deteksi wajah menggunakan detector, koordinat akan masuk kedalam variable rects
    rects = detector.detectMultiScale(frame, scaleFactor=1.1,
    # minSize 120, mendeteksi wajah paling jauh sekitar 3 meter dari kamera
    # Dipilih 3 meter karena lebih dari 3 meter pendeteksi landmark mulai tidak akurat (Banyak False Positive)
        minNeighbors=7, minSize=(120, 120),
        flags=cv2.CASCADE_SCALE_IMAGE)
    return rects