#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from PIL import Image
import math
import pytesseract

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg

OFFSET = 0 # Crop image offset to remove left over black boarder
AREA_CUTOFF = 200 # Contours must have greater than area cutoff (pixel^2) to be considered

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

def crop(frame, contour):
	rect = cv.minAreaRect(contour)
	box = cv.boxPoints(rect)
	box = np.int0(box)
	x,y,w,h = cv.boundingRect(box)
	cropped = frame[y+OFFSET:y-OFFSET+h, x+OFFSET:x-OFFSET+w]

	return cropped

def object_localization(binary):
    _, contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    localization = []
    for c in contours:
        cropped = None
        if check_square(c, 0.02):
            cropped = crop(binary, c)
            localization.append(cropped)
            cv.imshow('Cropped', cropped)
            if cropped is not None:
                cv.imshow('Cropped', cropped)

    return np.array(localization)

def detect_sign(image):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    except CvBridgeError as e:
        rospy.logWarn(e)

    binary = binary_classification(cv_image, 177)
    localization = object_localization(binary)
    for image in localization:
        input = Image.fromarray(image, mode="L")
        text = pytesseract.image_to_string(input, config="--psm 7")
        rospy.loginfo(text)

    cv.imshow('Vidoe', binary)
    cv.waitKey(1)



if __name__ == "__main__":
    rospy.init_node("OpticalCharecterRecognizer")
    rospy.Subscriber("/usb_cam/image_raw", Image_msg, detect_sign)
    # img = cv.imread("../images/Sign 75.png")
    #
    # binary = binary_classification(img, 177)
    # localization = object_localization(binary)
    # for image in localization:
    #     input = Image.fromarray(image, mode="L")
    #     text = pytesseract.image_to_string(input, config="--psm 7")
    #     print(text)

    rospy.on_shutdown(shutdown_hook)

    rospy.spin()
