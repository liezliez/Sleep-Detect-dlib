# import package
from imutils import face_utils
from imutils.video import VideoStream
import time as t
import dlib
import cv2
from hitungEAR import eye_aspect_ratio, nilai_ear
import numpy as np

# argument parser untuk model yang dipakai
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--sp", required=True,
# 	help="Model yang dipakai")
# args = vars(ap.parse_args())

# print("Memulai Program...")

# inisiasi RPI GPIO untuk relay
# import RPi.GPIO as GPIO
# import time

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(23, GPIO.OUT)
# GPIO.setup(24, GPIO.OUT)

# inisiasi face detector haarcascade opencv
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# inisiasi face detector bawaan dlib
# detector = dlib.get_frontal_face_detector()

# inisiasi shape predictor untuk facial landmark dari dlib
# predictor = dlib.shape_predictor(args["sp"])
predictor = dlib.shape_predictor("Model (4).dat")
# inisiasi video camera
vs = VideoStream(src=0).start()

# counter frame untuk pengguna dinyatakan tertidur
COUNTER_EAR = 30

# treshold ear untuk pengguna dinyatakan telah menutup matanya
EYE_AR_THRESH = 0.222

DETIK = 5
TIDAK_TERDETEKSI_TRESH = 20

# inisiasi nilai
counter_mati = 0
counter_frame = 0
avg = 0
nilai= 0
tidak_terdeteksi = False

start_time = t.time()

# loop frame dari videostream kamera, false condition = Q (Keyboard interupt)
try:
	while True:
		current_time = t.time()
		elapsed_time = current_time - start_time

	# read video input untuk dijadikan frame lalu convert grayscale
		frame = vs.read()
		# frame.flags.writeable = False
		frame.flags.writeable = True

	# convert grayscale opencv
		abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# deteksi wajah menggunakan detector, koordinat akan masuk kedalam variable rects
		rects = detector.detectMultiScale(abu, scaleFactor=1.1,
		# minSize 120, mendeteksi wajah paling jauh sekitar 3 meter dari kamera
		# lebih dari 3 meter, asepect ration mulai tidak akurat
			minNeighbors=7, minSize=(120, 120),
			flags=cv2.CASCADE_SCALE_IMAGE)

	# terdeteksi wajah (tuple null jika tidak terdeteksi wajah)
		if len(rects) != 0:

		# reset pada counter ketika wajah terdeteksi kembali
			tidak_terdeteksi = False
			counter_mati = 0

			for (x, y, w, h) in rects:
				rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

				# bikin bounding box face
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

			# deteksi landmark wajah, dirubah ke numpy
				shape = predictor(abu, rect)
				shape = face_utils.shape_to_np(shape)

			# hitung EAR
				eye = nilai_ear(shape)
				ear = eye[0]
				leftEye = eye [1]
				rightEye = eye[2]

				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 0)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 0)

			# hitung average ear dari waktu yang ditentukan
				counter_frame += 1
				nilai = nilai + ear
		
			# kondisi 2 (Terjaga)	
				cv2.putText(frame, "Nilai EAR : {:.2f}".format(ear), (300, 30),
										cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
			# kondisi 3 (Tidak Terdeteksi)
			else:
				text = "Tidak Terdeteksi"
				cv2.putText(frame, text, (250,250), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
				counter_mati += 1	
			
			cv2.putText(frame, "counter : {:.0f}".format(counter_mati), (300, 60),
								cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)

	# Trigger
	# jika waktu yang telah dihabiskan telah melebihi waktu DETIK untuk pengecekan, maka hitung Average dari nilai total EAR terhadap frame
		if elapsed_time > DETIK:
			try:
			# jika wajah tidak terdeteksi pada frame pertama
				avg = nilai/counter_frame
			except:
				print("Wajah tidak terdeteksi (average 0)")
			print(avg)
		# jika Average melebihi treshold, maka pengguna dinyatakan telah tertidur/meninggalkan alat (tidak terdeteksi sedang menggunakan alat)
			if avg < EYE_AR_THRESH :
				timestr = t.strftime("%Y%m%d-%H%M%S")
				print("MATI (EAR MELEBIHI TRESHOLD)")
				cv2.imwrite("./hasil/frame%s.jpg" % timestr, frame )
				# os.system("irsend SEND_ONCE --count=4 Sony_RM-ED035 KEY_SLEEP")
				# GPIO.output(23, False)
				# GPIO.output(23, True)
				input("Reset tekan enter")
				# os.system("irsend SEND_ONCE --count=4 Sony_RM-ED035 KEY_SLEEP")
				# GPIO.output(23, False)
				nilai = 0
				counter_frame = 1
				start_time = t.time()

			else :
				nilai = 0
				counter_frame = 1
				start_time = t.time()
				
	# tampilkan frame
		cv2.imshow("Kamera", frame)
		key = cv2.waitKey(1) & 0xFF
	# q keluar
		if key == ord("q"):
			break
		# time.sleep(0.5)
	# destroy
	cv2.destroyAllWindows()
	vs.stop()
except KeyboardInterrupt:
	pass