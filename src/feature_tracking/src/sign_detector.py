#!/usr/bin/env python

# Reference for HuMoments code: https://www.learnopencv.com/shape-matching-using-hu-moments-c-python/
import rospy
import cv2 as cv
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

# Hu Moments for each sign
Sign_Turn_Left = np.array([0.60051821, 1.77938078, 2.41080726, 3.57778346, -6.57837756, -4.47255285, -7.34397072])
Sign_Turn_Right = np.array([0.60582587, 1.82948018, 2.41594102, 3.60746856, -6.62084558, -4.55277538, -7.67673815])
Sign_25 = np.array([0.32807563, 1.2390984, 3.26559887, 3.90878974, 7.74434791, 5.08009115, -7.57928918])
Sign_75 = np.array([0.20836988, 0.79547836, 0.83454143, 1.53649249, 2.82159092, 1.98131614, -2.93918862])
Sign_100 = np.array([0.30514208, 0.72741868, 1.8562603, 2.32033954, 4.42215648, 2.6869574, -5.01829956])
Sign_Stop = np.array([0.38214421, 1.74043268, 3.61796505, 5.32510358, 10.03675349, 7.06024509, 9.88390894])

Sign_5Points_Star = np.array([0.66553197, 4.48304069, 5.66387012, 7.02322334, -13.3701022, 9.26550831, 0.24645356, 0.9929972])
Sign_8Points_Star = np.array([ 0.78778773, 5.03687163, 8.99199894, 12.67354382, -15.65355978, -15.64138739, 0.67429712, 0.99083969])
Sign_Arc = np.array([0.46503047, 1.22294326, 1.95947349,  2.875807, -5.29443138, -3.48758083, 0.41724003, 0.49579832])
Sign_Arrow = np.array([0.61444261, 1.56520217, 3.29744171, 4.31056882, 8.14039103, 5.11504075, 0.47413516, 0.77570093])
Sign_Heart = np.array([0.75666305, 2.8885595, 3.06542999, 6.47723769, -11.24933264, -7.92189042, 0.61244725, 0.99190939])
Sign_Octagon = np.array([0.7969718, 4.62385146, 15.65355977, 15.65355977, 15.65355977, 15.65355977, 0.94767216, 0.96436526])
Sign_Triangle = np.array([0.71526353, 4.13538047, 2.34029664, 5.62641892, 9.6097774, 7.69411097, 0.55059386, 0.90554415])



OFFSET = 0 # Crop image offset to remove left over black boarder
AREA_CUTOFF = 50 # Contours must have greater than area cutoff (pixel^2) to be considered

def shutdown_hook():
    cv.destroyAllWindows()

def binary_classification(frame, threshold):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _ , thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
  
    return thresh

def check_square(contour, tol):
	cir = calc_circularity(contour)
	target = math.pi/4.0

	diff = (cir - target)**2

	return diff <= tol 

def calc_circularity(contour):
	area = cv.contourArea(contour)
	parim = cv.arcLength(contour, closed=True)
	
	if area >= AREA_CUTOFF:
		circularity = 4*math.pi*area/float(parim**2)
	else:
		circularity = 999

	return circularity

def calc_aspectRatio(contour):
	rect = cv.minAreaRect(contour)
	box = cv.boxPoints(rect)
	box = np.int0(box)
	_,_,w,h = cv.boundingRect(box)
	
	return min((w,h))/float(max((w,h)))

def draw_contours(original, thresh):
	frame_copy = original.copy()
	features = []
	_, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	cropped = None
	for c in contours:
		if check_square(c, 0.02):
			cv.drawContours(frame_copy, c, -1, (0,255,0), 2)
			cropped, internal_contours = crop(thresh, c)
			features.append(get_feature(internal_contours))
			if cropped is not None:
				cv.imshow('Cropped', cropped)
				cv.waitKey(1)
			

	return frame_copy, cropped, features

def crop(frame, contour):
	rect = cv.minAreaRect(contour)
	box = cv.boxPoints(rect)
	box = np.int0(box)
	x,y,w,h = cv.boundingRect(box)
	cropped = frame[y+OFFSET:y-OFFSET+h, x+OFFSET:x-OFFSET+w]

	_, contours, hierarchy = cv.findContours(cropped, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	hierarchy = np.squeeze(hierarchy)

	cropped = cv.cvtColor(cropped, cv.COLOR_GRAY2BGR)
	internal_contours = []
	if len(hierarchy.shape) > 1:
		for i in range(hierarchy.shape[0]):
			if hierarchy[i][3] > -1 and cv.contourArea(contours[i]) >= AREA_CUTOFF:
				cv.drawContours(cropped, contours[i], -1, (0,255,0), 2)
				internal_contours.append(contours[i])

	return cropped, internal_contours


def get_feature(contours):
	features = []
	for cnt in contours:
		moment = cv.moments(cnt)
		huMoment = np.squeeze(cv.HuMoments(moment))

		for i in range(len(huMoment)):
			huMoment[i] = -1* math.copysign(1.0, huMoment[i]) * math.log10(abs(huMoment[i]+np.finfo(float).eps))
		feature = huMoment[:-1]
		feature = np.append(feature, calc_circularity(cnt))
		feature = np.append(feature, calc_aspectRatio(cnt))
		features.append(feature)

	return features


def detect_sign(image):
	bridge = CvBridge()
	try:
		cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
	except CvBridgeError as e:
		rospy.logWarn(e)

	binary = binary_classification(cv_image, 177)
	drawn, _, blobs = draw_contours(cv_image, binary)
	
	# Brute-Force Feature Matching
	candidates = []
	for blob in blobs:
		if len(blob) <= 6:
			dist = []
			for ft in blob:
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_5Points_Star))))
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_8Points_Star))))
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_Arc))))
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_Arrow))))
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_Heart))))
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_Octagon))))
				dist.append(np.sqrt(np.sum(np.square(ft - Sign_Triangle))))
			if len(dist) > 0:
				dist = np.array(dist)

				index = np.argmin(dist)
				candidates.append((dist[index], index))


	if len(candidates) > 0:
		result = min(candidates, key=lambda x: x[0])
		signs = ["5 Points Star","8 Points Star","Arc","Arrow","Heart","Octagon", "Triangle"]
		if result[0] <= 1:
			rospy.loginfo("Math Found! It's sign " + signs[result[1]%8])
		else:
			rospy.loginfo(result)

	
	cv.imshow('Vidoe/Debug', np.hstack((drawn, cv.cvtColor(binary, cv.COLOR_GRAY2BGR))))
	cv.waitKey(1)


if __name__ == "__main__":
	rospy.init_node("Sign_Detector")

	rospy.Subscriber("/camera/rgb/image_raw", Image, detect_sign)

	rospy.on_shutdown(shutdown_hook)

	rospy.spin()
