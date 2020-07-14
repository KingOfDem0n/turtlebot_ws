#!/usr/bin/env python

import cv2 as cv
import numpy as np
import math

OFFSET = 0 # Crop image offset to remove left over black boarder

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
	
	for i in range(hierarchy.shape[0]):
		if hierarchy[i][3] > -1:
			cv.drawContours(cropped, contours[i], -1, (0,255,0), 2)
			internal_contours.append(contours[i])
			cv.imshow('image', cropped)
			cv.waitKey(0)

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


if __name__ == "__main__":
	img = cv.imread("../images/Sign_Stop.png")
	thresh = binary_classification(img, 177)
	drawn, cropped, huMoments = draw_contours(img, thresh)
	print(huMoments[0])

	cv.imshow('image', cropped)
	cv.waitKey(0)
	cv.destroyAllWindows()
