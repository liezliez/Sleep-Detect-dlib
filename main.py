# import package
from imutils import face_utils
from imutils.video import VideoStream
import argparse
import imutils
import time
import dlib
import cv2
from hitungEAR import eye_aspect_ratio, final_ear

# argument parser untuk model yang dipakai
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--sp", required=True,
	help="Model yang dipakai")
args = vars(ap.parse_args())

print("Memulai Program...")

# inisiasi face detector
# inisiasi face detector haarcascade opencv
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# inisiasi face detector bawaan dlib
# detector = dlib.get_frontal_face_detector()

# inisiasi shape predictor untuk facial landmark dari dlib
predictor = dlib.shape_predictor(args["sp"])

# inisiasi video camera
vs = VideoStream(src=0).start()

# counter frame untuk pengguna dinyatakan tertidur
COUNTER_EAR = 30

# treshold ear untuk pengguna dinyatakan telah menutup matanya
EYE_AR_THRESH = 0.25

# inisiasi counter
COUNTER = 0

# loop frame dari videostream kamera, false condition = Q (Keyboard interupt)
while True:
  # read video input untuk dijadikan frame lalu convert grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=720)

  # convert grayscale opencv
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # deteksi wajah menggunakan detector, koordinat akan masuk kedalam variable rects
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	checkTuple = len(rects)

  # terdeteksi wajah (tuple null jika tidak terdeteksi wajah)
	if checkTuple != 0:
		for (x, y, w, h) in rects:
			rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

			# bikin bounding box face
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

		# deteksi landmark wajah, dirubah ke numpy
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

		# kondisi 1 (Tertidur)
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				
		# kondisi 2 (Terjaga)	
			else:
				COUNTER = 0
			
			cv2.putText(frame, "Nilai EAR : {:.2f}".format(ear), (500, 30),
						cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
	# kondisi 3 (Tidak Terdeteksi)
	else:
		cv2.putText(frame, "Tidak Terdeteksi", (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		COUNTER += 1
	
	cv2.putText(frame, "counter : {:.2f}".format(COUNTER), (500, 60),
						cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
	print(COUNTER)
	if COUNTER >= COUNTER_EAR:
					cv2.putText(frame, "Tertidur", (200, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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