#https://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
#https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/

from threading import Thread
import time
import cv2
import dlib
import numpy as np
from multiprocessing import Process, Queue
import queue

draw_freq = 3
fac = 0.5
src = 0
predictor_path = 'd:/data/faces/shape_predictor_68_face_landmarks.dat'

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

class Detect:
	def __init__(self):
		self.land = []
		self.frame_q = Queue(1)
		self.land_q = Queue(1)
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(predictor_path)
		t = Process(target=self.update, args=())
		t.daemon = True
		t.start()

	def get_land(self, img):
		dets = self.detector(img, 0)
		shapes = []
		for k, d in enumerate(dets):
			p = self.predictor(img, d)

			p = shape_to_np(p)
			shapes.append(p)
		return shapes

	def update(self):
		while True:
			try:
				frame = self.frame_q.get_nowait()
			except queue.Empty:
				continue
			land = self.get_land(frame)
			try:
				self.land_q.put_nowait(land)
			except queue.Full:
				try:
					self.land_q.get_nowait()
				except queue.Empty:
					pass
				self.land_q.put(land)

	def read(self, frame, fac=1):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if fac != 1:
			frame = cv2.resize(frame, None, fx=fac, fy=fac)
		try:
			self.frame_q.put_nowait(frame)
		except queue.Full:
			try:
				self.frame_q.get_nowait()
			except queue.Empty:
				pass
			self.frame_q.put(frame)
		try:
			self.land = self.land_q.get_nowait()
		except queue.Empty:
			pass
		return self.land

class Camera:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		self.grabbed, self.frame = self.stream.read()
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()

	def update(self):
		while True:
			self.grabbed, frame = self.stream.read()
			self.frame = cv2.flip(frame, 1)

	def read(self):
		return self.frame

	def stop(self):
		self.stream.release()

def draw_polyline(img, d, start, end, fac=1, closed=False):
	pts = [np.asarray([[[int(x / fac), int(y / fac)]] for x, y in d[start:end + 1]])]
	cv2.polylines(img, pts, closed, (255, 0, 0), 1)

def draw_face(img, d, fac):
	draw_polyline(img, d, 0, 16, fac) # Jaw line
	draw_polyline(img, d, 17, 21, fac) # Left eyebrow
	draw_polyline(img, d, 22, 26, fac) # Right eyebrow
	draw_polyline(img, d, 27, 30, fac) # Nose bridge
	draw_polyline(img, d, 30, 35, fac, True) # Lower nose
	draw_polyline(img, d, 36, 41, fac, True) # Left eye
	draw_polyline(img, d, 42, 47, fac, True) # Right Eye
	draw_polyline(img, d, 48, 59, fac, True) # Outer lip
	draw_polyline(img, d, 60, 67, fac, True) # Inner lip

if __name__ == '__main__':
	frame_cnt = 0
	cam = Camera(src)
	det = Detect()

	start = time.time()
	while True:
		frame = cam.read()
		if frame is None:
			break

		faces = det.read(frame, fac)
		if frame_cnt % draw_freq == 0:
			for p in faces:
				draw_face(frame, p, fac)
				#for (x, y) in p:
				#	cv2.circle(frame, (int(x / fac), int(y / fac)), 1, (255, 0, 0), -1)

		cv2.imshow('', frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break

		frame_cnt += 1

	print("FPS=%.2f" % (frame_cnt/(time.time()-start)))
	cv2.destroyAllWindows()
	cam.stop()