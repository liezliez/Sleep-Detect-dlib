import cv2
from imutils import face_utils
import dlib

predictor = dlib.shape_predictor("Model (4).dat")

def deteksiLandmark(rects,frame):
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

            # bikin bounding box face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # deteksi landmark wajah, dirubah ke numpy
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)
        return shape