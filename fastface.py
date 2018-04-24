# see also:
# https://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/

from threading import Thread
import time
import cv2
import dlib
import numpy as np
from multiprocessing import Process, Value
import argparse

cam_src = 0
mirror = True
scale_frac = 0.5
show_face = True
show_lines = True
show_marks = False
line_color = (255, 0, 0)
mark_color = (255,0,192)
predictor_path = 'd:/data/faces/shape_predictor_68_face_landmarks.dat'

class Display:
	def __init__(self, cam, det):
		self.terminated = False
		t = Thread(target=self.update, args=(cam,det))
		t.daemon = True
		t.start()

	def update(self, cam, det):
		frame_cnt = 0
		cur_frame = None
		while cam.frame is None:
			pass
		start = time.time()
		while not self.terminated:
			if show_face:
				cur_frame = cam.frame
			elif cur_frame is None:
				cur_frame = np.zeros_like(cam.frame)

			if show_lines or show_marks:
				cur_faces = det.faces
				for face in cur_faces:
					if show_lines:
						cv2.polylines(cur_frame, [face[:17]], False, line_color)  # Jaw line
						cv2.polylines(cur_frame, [face[17:22]], False, line_color)  # Left eyebrow
						cv2.polylines(cur_frame, [face[22:27]], False, line_color)  # Right eyebrow
						cv2.polylines(cur_frame, [face[27:31]], False, line_color)  # Nose bridge
						cv2.polylines(cur_frame, [face[30:36]], True, line_color)  # Lower nose
						cv2.polylines(cur_frame, [face[36:42]], True, line_color)  # Left eye
						cv2.polylines(cur_frame, [face[42:48]], True, line_color)  # Right Eye
						cv2.polylines(cur_frame, [face[48:60]], True, line_color)  # Outer lip
						cv2.polylines(cur_frame, [face[60:68]], True, line_color)  # Inner lip

					if show_marks:
						for pnt in face:
							cv2.circle(cur_frame, tuple(pnt[0]), 1, mark_color, -1)

			cv2.imshow('', cur_frame)
			frame_cnt += 1
			if cv2.waitKey(1) & 0xFF == 27:
				break

		delta_t = time.time() - start
		if delta_t > 0:
			print('FPS=%.2f' % (frame_cnt / delta_t))
		self.terminate()

	def terminate(self):
		if not self.terminated:
			cv2.destroyAllWindows()
		self.terminated = True


class Camera:
	def __init__(self, src=0, mirror=False):
		self.src = src
		self.mirror = mirror
		self.stream = cv2.VideoCapture(src)
		if not self.stream.isOpened():
			print('Error: camera %d not available' % self.src)
			self.terminated = True
			return -1
		self.terminated = False
		self.read()
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()

	def update(self):
		while not self.terminated:
			self.read()

	def read(self):
		grabbed, frame = self.stream.read()
		if not grabbed or frame is None:
			print('Error: camera %d not available'%self.src)
			self.terminate()
		elif self.mirror:
			self.frame = cv2.flip(frame, 1)
		else:
			self.frame = frame

	def terminate(self):
		if not self.terminated:
			self.stream.release()
		self.terminated = True

class Detect:
	def __init__(self, cam_src=0, mirror=False):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(predictor_path)
		self.faces = []
		self.cam = None
		self.disp = None
		self.terminated = Value('b', False)
		self.external_terminate_signal = Value('b', False)
		t = Process(target=self.update, args=(cam_src, mirror))
		t.daemon = True
		t.start()

	def update(self, cam_src=0, mirror=False):
		self.cam = Camera(cam_src, mirror)
		self.disp = Display(self.cam, self)
		while not self.terminated.value:
			if self.external_terminate_signal.value or (self.cam is not None and self.cam.terminated) or (self.disp is not None and self.disp.terminated):
				if self.cam is not None:
					self.cam.terminate()
				if self.disp is not None:
					self.disp.terminate()
				self.terminated.value = True
				break

			preproc_frame = cv2.cvtColor(self.cam.frame, cv2.COLOR_BGR2GRAY)
			if scale_frac != 1:
				preproc_frame = cv2.resize(preproc_frame, None, fx=scale_frac, fy=scale_frac)
			self.faces = [(np.array([[[p.x, p.y]] for p in self.predictor(preproc_frame, rect).parts()])/scale_frac).astype(int) for rect in self.detector(preproc_frame, 0)]

	def is_terminated(self):
		return self.terminated.value

	def terminate(self):
		self.external_terminate_signal.value = True


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--src', type=int, help='camera source', default=cam_src)
	parser.add_argument('--mirror', type=int, help='mirror camera', default=mirror)
	args = parser.parse_args()

	det = Detect(args.src, args.mirror)
	while not det.is_terminated():
		pass
