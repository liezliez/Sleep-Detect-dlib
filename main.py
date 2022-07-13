# -------------------- import package --------------------

from imutils.video import VideoStream
import time as t
import cv2
import logging

from hitungEAR import nilai_ear
from deteksiWajah import deteksiWajah
from deteksiLandmark import deteksiLandmark
# Untuk di Raspberry Pi
# from aktuator import hidupkan, matikan

# -------------------- Program dimulai --------------------
print("Memulai Program...")

# -------------------- logger --------------------
lgr = logging.getLogger('Main.py')
lgr.setLevel(logging.DEBUG) # log all escalated at and above DEBUG

# add a file handler
fh = logging.FileHandler('./hasil/log-tertidur.csv')
fh.setLevel(logging.DEBUG) # ensure all messages are logged to file
frmt = logging.Formatter('%(asctime)s,%(message)s')
fh.setFormatter(frmt)

# add the Handler to the logger
lgr.addHandler(fh)

# -------------------- inisiasi camera --------------------
vs = VideoStream(src=0).start()

# -------------------- constant --------------------

# treshold ear untuk pengguna dinyatakan telah menutup matanya
EAR_Treshold = 0.235
WAKTU_PENGECEKAN = 5

# inisiasi nilai
counter_frame = 0
avg_ear = 0
total_ear = 0

start_time = t.time()

# loop frame dari videostream kamera, false condition = Q (Keyboard interupt)
try:
	while True:
		current_time = t.time()
		elapsed_time = current_time - start_time

	# read video input, ubah ke grayscale
		frame = vs.read()
		frame_abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# -------------------- deteksi wajah --------------------
		rects = deteksiWajah(frame_abu)

	# -------------------- jika terdeteksi wajah (tuple tidak null) --------------------
		if len(rects) != 0:

		# -------------------- deteksi landmark --------------------
			shape = deteksiLandmark(rects, frame_abu)

		# hitung EAR
			eye = nilai_ear(shape)

		# untuk GUI openCV

			ear = eye[0]
			leftEye = eye [1]
			rightEye = eye[2]

		# tampilkan hasil deteksi di GUI
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 0)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 0)

		# hitung average ear dari waktu yang ditentukan
			counter_frame += 1
			total_ear = total_ear + ear

		# tampilkan hasil landmark di GUI
			cv2.putText(frame, "Nilai EAR : {:.2f}".format(ear), (300, 30),
									cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
		
	# -------------------- jika wajah tidak terdeteksi --------------------
		else:

		# tampilkan text "tidak terdeteksi" GUI
			text = "Tidak Terdeteksi"
			cv2.putText(frame, text, (250,250), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

	# tampilkan nilai rata-rata EAR dan elapsed time di GUI

		cv2.putText(frame, "Nilai AVG : {:.2f}".format(avg_ear), (300, 60),
			cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "elapsed time : {:.2f}".format(elapsed_time), (300, 90),
			cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)
	# -------------------- Trigger --------------------

	# jika waktu yang telah dihabiskan telah melebihi waktu pengecekan, maka hitung rata-rata dari nilai total EAR terhadap frame

		if elapsed_time > WAKTU_PENGECEKAN:
		
		# Console feedback
			print("Avg EAR : ",avg_ear, " - Counter frame",counter_frame)

		# Error handling frame pertama
			try:
				avg_ear = total_ear/counter_frame
			except:
			# jika wajah tidak terdeteksi pada frame pertama
				print("Wajah tidak terdeteksi (EAR 0)")

			

		# jika Average melebihi treshold, maka pengguna dinyatakan telah tertidur/meninggalkan alat
			if avg_ear < EAR_Treshold :
			# console print MATI
				print("MATI")

			# catat waktu dinyatakan tertidur
				timestr = t.strftime("%Y%m%d %H%M%S")
				cv2.putText(frame, "DIMATIKAN", (250, 250),
					cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 255), 2)

			# simpan frame terakhir ketika sistem dimatikan
				cv2.imwrite("./hasil/frame%s.jpg" % timestr, frame )

			# Input nilai rata-rata EAR terakhir ke logger
				lgr.info(avg_ear)
			

			# Matikan alat elektronik
				# matikan()

			# menunggu masukan user untuk mereset sistem
				input("Reset tekan enter")

			# Hidupkan kembali alat elektronik
				# hidupkan()

			# Atur ulang variabel inisiasi
				total_ear = 0
				counter_frame = 1
				start_time = t.time()

			else :
				total_ear = 0
				counter_frame = 1
				start_time = t.time()
				
	# OpenCV menampilkan frame pada window GUI
		cv2.imshow("Kamera", frame)
		key = cv2.waitKey(1) & 0xFF
	# key q keluar dari window
		if key == ord("q"):
			break
		# time.sleep(0.5)
# tutup program dan matikan kamera
	cv2.destroyAllWindows()
	vs.stop()
# exit keyboard interrupt (CTRL + C)
except KeyboardInterrupt:
	pass