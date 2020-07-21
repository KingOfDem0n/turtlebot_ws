#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from PIL import Image
import math
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

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
            # if cropped is not None:
            #     cv.imshow('Cropped', cropped)
            #     cv.waitKey(1)

    return np.array(localization)

# def detect_sign(image):
#     bridge = CvBridge()
# 	try:
# 		cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
# 	except CvBridgeError as e:
# 		rospy.logWarn(e)
#
#     transformer = transforms.Compose([transforms.Resize((224,224)),
#                                       transforms.Grayscale(3),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
#     binary = binary_classification(cv_image, 177)
#     localization = object_localization(binary)
#     localization = transformer(localization)
#
#     with torch.no_grad():
#         localization.to(device)
#
#         network()



if __name__ == "__main__":
    rospy.init_node("Feature_tracking_DeepNetwork")
    img = cv.imread("../images/Sign_Triangle.png")
    classes = ['5PointStar',
             '8PointStar',
             'Arc',
             'Arrow',
             'Heart',
             'LeftTurn',
             'Octagon',
             'Others',
             'RightTurn',
             'Triangle']

    # Establish network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = models.resnet18(pretrained=True)

    # Replace last layer
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 10)
    network.load_state_dict(torch.load("../models/first.pth", map_location=device))
    network.to(device)
    network.eval()

    transformer = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224,224)),
                                      transforms.Grayscale(3),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    binary = binary_classification(img, 177)
    localization = object_localization(binary)
    localization = transformer(localization)

    with torch.no_grad():
        localization.to(device)
        outputs = network(torch.unsqueeze(localization, dim=0))
        _, preds = torch.max(outputs, 1)
        print(outputs)
        print(classes[preds.item()])

    cv.waitKey(0)

    # rospy.Subscriber("/usb_cam/image_raw", Image, detect_sign)

    rospy.on_shutdown(shutdown_hook)

	# rospy.spin()
