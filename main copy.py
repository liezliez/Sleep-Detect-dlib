# import package
from audioop import avg
from imutils import face_utils
from imutils.video import VideoStream
import argparse
import imutils
import time as t
import dlib
import cv2
import mediapipe as mp
import numpy as np
from hitungEAR import eye_aspect_ratio, final_ear

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# argument parser untuk model yang dipakai
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--sp", required=True,
	help="Model yang dipakai")
args = vars(ap.parse_args())

print("Memulai Program...")

# inisiasi face detector
# inisiasi face detector haarcascade opencv
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# inisiasi shape predictor dari model
predictor = dlib.shape_predictor(args["sp"])

# inisiasi video camera
vs = VideoStream(src=0).start()

# counter frame untuk pengguna dinyatakan tertidur
COUNTER_EAR = 30
# treshold ear untuk pengguna dinyatakan telah menutup matanya
EYE_AR_THRESH = 0.23
# inisiasi counter
COUNTER = 0

counter_frame = 0
avg = 0
nilai= 0
detik = 10

start_time = t.time()

# loop frame dari videostream kamera, false condition = Q (Keyboard interupt)
while True:
  # read video input untuk dijadikan frame lalu convert grayscale

	current_time = t.time()
	elapsed_time = current_time - start_time

	frame = vs.read()

	frame.flags.writeable = False
	results = face_mesh.process(frame)
	frame.flags.writeable = True


  # convert grayscale opencv
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	face_3d = []
	face_2d = []
	img_h, img_w, img_c = frame.shape

	# deteksi wajah menggunakan detector, koordinat akan masuk kedalam variable rects
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=6, minSize=(90, 90),
		flags=cv2.CASCADE_SCALE_IMAGE)
	checkTuple = len(rects)
	# terdeteksi wajah (tuple null jika tidak terdeteksi wajah)

	# head pose estimation
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			for idx, lm in enumerate(face_landmarks.landmark):
				if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
					if idx == 1:
						nose_2d = (lm.x * img_w, lm.y * img_h)
						nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

					x, y = int(lm.x * img_w), int(lm.y * img_h)
					face_2d.append([x, y])
					face_3d.append([x, y, lm.z]) 
			face_2d = np.array(face_2d, dtype=np.float64)
			face_3d = np.array(face_3d, dtype=np.float64)

			focal_length = 1 * img_w

			cam_matrix = np.array([ [focal_length, 0, img_h / 2],
									[0, focal_length, img_w / 2],
									[0, 0, 1]])

			dist_matrix = np.zeros((4, 1), dtype=np.float64)
			success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
			rmat, jac = cv2.Rodrigues(rot_vec)
			angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

			x = angles[0] * 360
			y = angles[1] * 360

			if y < -12:
				text = "Looking Right"
				COUNTER += 1
			elif y > 12:
				text = "Looking Left"
				COUNTER += 1
			elif x < -9:
				text = "Looking Down"
				COUNTER += 1
			else:
				text = "Forward"
				if checkTuple != 0:
					# hitung ear
					for (x, y, w, h) in rects:
						rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

						# bikin bounding box face
						(x, y, w, h) = face_utils.rect_to_bb(rect)
						cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

					# landmark wajah dirubah ke numpy
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
						# nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
						# p1 = (int(nose_2d[0]), int(nose_2d[1]))
						# p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
						# cv2.line(frame, p1, p2, (255, 0, 0), 2)

						counter_frame += 1
						nilai = nilai+ear
						
					# kondisi 1 (Tertidur)
						if ear < EYE_AR_THRESH:
							COUNTER += 1
					# kondisi 2 (Terjaga)	
						else:
							COUNTER = 0
						cv2.putText(frame, "Nilai EAR : {:.2f}".format(ear), (300, 30),
									cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
				else:
					COUNTER += 1
	else:
		cv2.putText(frame, "Tidak Terdeteksi", (700, 80),
			cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
		text = "tidak terdeteksi"
		COUNTER += 1

	cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "counter : {:.0f}".format(COUNTER), (300, 60),
						cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
	if COUNTER >= COUNTER_EAR:
					cv2.putText(frame, "Tertidur/Tidak Terdeteksi", (800, 30),
								cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)


	if elapsed_time > detik:
		avg = nilai/counter_frame
		print(avg)
		if avg < 0.23 :
			print("MATI DAH")
			nilai = 0
			counter_frame = 1
			start_time = t.time()
		else :
			nilai = 0
			counter_frame = 1
			start_time = t.time()

	

  # frame
	cv2.imshow("Sleep Detection", frame)
	key = cv2.waitKey(1) & 0xFF
  # q keluar
	if key == ord("q"):
		break
	# time.sleep(0.5)
# destroy
cv2.destroyAllWindows()
vs.stop()