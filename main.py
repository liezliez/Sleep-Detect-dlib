# library
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist

# argument parser untuk command line shape-predictor yang dipakai
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--sp", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# fungsi hitung EAR

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
	#Full landmark koordinat
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

	# custom model
    # leftEye = shape[0:6]
    # rightEye = shape[6:12]

	

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

print("[INFO] inisiasi Program...")

# inisiasi face detector dari dlib
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["sp"])

# inisiasi video camera

vs = VideoStream(src=0).start()

# vs = VideoStream(src)
# time.sleep(2.0)

COUNTER_EAR = 30
EYE_AR_THRESH = 0.27
COUNTER = 0

# cap = cv2.VideoCapture('test.mp4')


# loop frame dari videostream kamera, false condition = q ditekan
while True:
  # video input , convert grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=720)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # deteksi wajah
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	checkTuple = len(rects)

	# terdeteksi wajah
	if checkTuple != 0:
		for (x, y, w, h) in rects:
			rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

		# deteksi landmark wajah
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

		# hitung EAR
			eye = final_ear(shape)
			ear = eye[0]
			leftEye = eye [1]
			rightEye = eye[2]

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 0)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 0)
			print(COUNTER)

		# kondisi 1 (Tertidur)
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				if COUNTER >= COUNTER_EAR:
					cv2.putText(frame, "Tertidur", (10, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# kondisi 2 (Terjaga)	
			else:
				COUNTER = 0
			
			cv2.putText(frame, "Nilai EAR : {:.2f}".format(ear), (300, 30),
						cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
	# kondisi 3 (Tidak Terdeteksi)
	else:
		cv2.putText(frame, "Tidak Terdeteksi", (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		COUNTER += 1

#   # loop face detection
# 	for rect in rects:

# 		# bikin bounding box
# 		(x, y, w, h) = face_utils.rect_to_bb(rect)
# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
# 	# dari dlib shape predictor, masukin koordinat facial landmark ke NumPy array	
# 		shape = predictor(gray, rect)
# 		shape = face_utils.shape_to_np(shape)
		
#   # loop untuk dlib shape
# 		for (sX, sY) in shape:
# 			cv2.circle(frame, (sX, sY), 1, (0, 255, 0), -1)
  # frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
  # q keluar
	if key == ord("q"):
		break
	# time.sleep(0.5)
# destroy
cv2.destroyAllWindows()
vs.stop()