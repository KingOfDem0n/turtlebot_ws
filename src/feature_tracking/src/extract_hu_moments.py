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

def calc_aspectRatio(contour):
	rect = cv.minAreaRect(contour)
	box = cv.boxPoints(rect)
	box = np.int0(box)
	_,_,w,h = cv.boundingRect(box)

	return min((w,h))/float(max((w,h)))

def calc_InterestPoint(cropped):
	gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv.cornerHarris(gray, 2, 3, 0.04)
	dst = cv.dilate(dst,None)
	cropped[dst>0.01*dst.max()]=[0,0,255]
	return cropped


def draw_contours(original, thresh):
	frame_copy = original.copy()
	features = []
	_, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	cropped = None
	for c in contours:
		if check_square(c, 0.01):
			cv.drawContours(frame_copy, c, -1, (0,255,0), 2)
			cropped, internal_contours = crop(thresh, c)
			features.append(get_feature(internal_contours))


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

	for i in range(hierarchy.shape[0]):
		if hierarchy[i][3] > -1:
			cv.drawContours(cropped, contours[i], -1, (0,255,0), 2)
			internal_contours.append(contours[i])
			cv.imshow('image', cropped)
			cv.waitKey(0)

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


if __name__ == "__main__":
	img = cv.imread("../images/Sign 25.png")
	thresh = binary_classification(img, 177)
	drawn, cropped, features = draw_contours(img, thresh)
	# dst = calc_InterestPoint(cropped)
	print(features[0])

	cv.imshow('image', cropped)
	cv.waitKey(0)
	cv.destroyAllWindows()
