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



OFFSET = 0 # Crop image offset to remove left over black boarder

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
	
	if area >= 50:
		circularity = 4*math.pi*area/float(parim**2)
	else:
		circularity = 999

	return circularity


def draw_contours(original, thresh):
	frame_copy = original.copy()
	huMoments = []
	_, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	cropped = None
	for c in contours:
		if check_square(c, 0.01):
			cv.drawContours(frame_copy, c, -1, (0,255,0), 2)
			cropped, contours = crop(thresh, c)
			huMoments.append(get_huMoment(contours))
			

	return frame_copy, cropped, huMoments

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
			if hierarchy[i][3] > -1:
				cv.drawContours(cropped, contours[i], -1, (0,255,0), 2)
				internal_contours.append(contours[i])

	return cropped, internal_contours


def get_huMoment(contours):
	huMoments = []
	for cnt in contours:
		moment = cv.moments(cnt)
		huMoment = np.squeeze(cv.HuMoments(moment))

		for i in range(len(huMoment)):
			huMoment[i] = -1* math.copysign(1.0, huMoment[i]) * math.log10(abs(huMoment[i]+np.finfo(float).eps))
		
		huMoments.append(huMoment)

	return huMoments

def find_mostLikelyMatch(hm):
	sign_25 

def detect_sign(image):
	bridge = CvBridge()
	try:
		cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
	except CvBridgeError as e:
		rospy.logWarn(e)

	binary = binary_classification(cv_image, 177)
	drawn, cropped, blobs = draw_contours(cv_image, binary)
	
	# Brute-Force Feature Matching
	candidates = []
	for blob in blobs:
		if len(blob) <= 6:
			for hm in blob:
				sign_25 = np.sqrt(np.sum(np.square(hm - Sign_25)))
				sign_75 = np.sqrt(np.sum(np.square(hm - Sign_75)))
				sign_100 = np.sqrt(np.sum(np.square(hm - Sign_100)))
				sign_Stop = np.sqrt(np.sum(np.square(hm - Sign_Stop)))
				sign_Turn_Left = np.sqrt(np.sum(np.square(hm - Sign_Turn_Left)))
				sign_Turn_Right = np.sqrt(np.sum(np.square(hm - Sign_Turn_Right)))
			
				dist = np.array([sign_25, sign_75, sign_100, sign_Stop, sign_Turn_Left, sign_Turn_Right])

				index = np.argmin(dist)
				candidates.append((dist[index], index))


	if len(candidates) > 0:
		result = min(candidates, key=lambda x: x[0])
		signs = ["25","75","100","Stop","Turn Left","Turn Right"]
		if result[0] <= 1:
			rospy.loginfo("Math Found! It's sign " + signs[result[1]])

	
	cv.imshow('Vidoe/Debug', np.hstack((drawn, cv.cvtColor(binary, cv.COLOR_GRAY2BGR))))
	cv.waitKey(1)


if __name__ == "__main__":
	rospy.init_node("Sign_Detector")

	rospy.Subscriber("/camera/rgb/image_raw", Image, detect_sign)

	rospy.on_shutdown(shutdown_hook)

	rospy.spin()
