#!/usr/bin/env python

# Reference for HuMoments code: https://www.learnopencv.com/shape-matching-using-hu-moments-c-python/
import rospy
import cv2 as cv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
from math import *

from dynamic_reconfigure.server import Server
from feature_tracking.cfg import Feature_trackingConfig

# TODO: Create a fucntion to merge overlapping boxes together

def shutdown_hook():
    global CTRL_C, cap

    CTRL_C = True
    cap.release()
    cv.destroyAllWindows()

def param_update(config, level):
    global bi_threshold, median_kernel

    bi_threshold = config["bi_threshold"]
    median_kernel = config["median_kernel"]

    return config

def color_classification(frame):
    frame_copy = frame.copy()
    hsv = cv.cvtColor(frame_copy, cv.COLOR_BGR2HSV)

    lower = np.array([30, 80, 80]) # 25, 52, 72
    upper = np.array([60, 255, 255])

    mask = cv.inRange(hsv, lower, upper)

    out = cv.bitwise_and(frame, frame, mask=mask)

    bgr = cv.cvtColor(out, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    _ , thresh = cv.threshold(gray, 60, 255, cv.THRESH_BINARY)

    blurred = filter(thresh)

    return blurred

def binary_classification(frame, threshold):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _ , thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    filtered = filter(thresh)

    return filtered

def filter(frame):
    global median_kernel

    kernel = np.ones((5,5), np.uint8)
    open = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    blurred = cv.medianBlur(open, int((median_kernel+1)*2+1))

    return blurred

def image_representation(frame):
    _, labels = cv.connectedComponents(frame)
    labels = np.uint8(labels)

    return labels

def calculate_center(moment):
    try:
        center_x = int(round(moment["m10"]/moment["m00"]))
        center_y = int(round(moment["m01"]/moment["m00"]))
    except:
        center_x = -1
        center_y = -1
    return (center_x, center_y)

def feature_extraction(labels):
    _, contours, _ = cv.findContours(labels, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    moments = map(cv.moments, contours)
    centers = map(calculate_center, moments)
    huMoments = map(cv.HuMoments, moments)

    for hm in huMoments:
        for i in range(0,7):
            hm[i] = -1* copysign(1.0, hm[i]) * log10(abs(hm[i]+np.finfo(float).eps))

    return centers, huMoments, contours

def draw_boxes(frame, centers, huMoments, contours):
    frame_copy = frame.copy()

    combined = sorted(zip(centers, contours, huMoments), key=lambda x: cv.contourArea(x[1]), reverse=True)

    # i = 0
    # while i < len(combined)-1:
    #     offset = 0
    #     x1,y1,w1,h1 = cv.boundingRect(combined[i][1])
    #     x1 = int(x1*(0.9))
    #     y1 = int(y1*(0.9))
    #     w1 = int(w1*(1.3))
    #     h1 = int(h1*(1.3))
    #     polygon = Polygon([(x1, y1), (x1, y1+h1), (x1+w1, y1+h1), (x1+w1, y1)])
    #     for j in range(i, len(combined)):
    #         center = combined[0]
    #         x2,y2,w2,h2 = cv.boundingRect(combined[j-offset][1])
    #         x2 = int(x2*(0.9))
    #         y2 = int(y2*(0.9))
    #         w2 = int(w2*(1.3))
    #         h2 = int(h2*(1.3))
    #         points = []
    #         points.append(Point(center[0],center[1]))
    #         points.append(Point(x2,y2))
    #         points.append(Point(x2+w2,y2))
    #         points.append(Point(x2+w2,y2+h2))
    #         points.append(Point(x2,y2+h2))
    #
    #         if polygon.contains(points[0]):
    #             combined.pop(j-offset)
    #             offset += 1
    #
    #         # for point in points:
    #         #     if polygon.contains(point):
    #         #         combined.pop(j-offset)
    #         #         offset += 1
    #         #         break
    #     i += 1

    for center, c, hu in combined:
        x,y,w,h = cv.boundingRect(c)
        x = int(x*(0.9))
        y = int(y*(0.9))
        w = int(w*(1.3))
        h = int(h*(1.3))
        cv.rectangle(frame_copy,(x,y),(x+w,y+h),(0,255,0),2)
        output_str = str(round(hu[0],2)) + "|" + str(round(hu[1],2)) + "|" + str(round(hu[2],2))
        cv.putText(frame_copy, output_str, center, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame_copy

def draw_contours(frame, centers, huMoments, contours):
    frame_copy = frame.copy()

    # Draw contours and mark center dot
    result = cv.drawContours(frame_copy, contours, -1, (0, 255, 0), 3)
    for center, p in zip(centers, huMoments):
        if center[0] != -1:
            result = cv.circle(result, center, 5, (255,0,0), -1)
            output_str = str(round(p[0],2)) + "|" + str(round(p[1],2)) + "|" + str(round(p[2],2))
            result = cv.putText(result, output_str, center, cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return result

if __name__ == "__main__":
    rospy.init_node("Feature_tracking")

    # Initialized variables
    CTRL_C = False
    rate = rospy.Rate(60)
    bi_threshold = 127
    median_kernel = 6

    # Open camera
    cap = cv.VideoCapture(1)
    if not (cap.isOpened()):
        rospy.logwarn("Can't open camera 1, defaulting back to camera 0")
        cap = cv.VideoCapture(0)
        if not (cap.isOpened()):
            rospy.logfatal("Can't open camera 0. Shuting down...")
            exit()

    rospy.on_shutdown(shutdown_hook)

    srv = Server(Feature_trackingConfig, param_update)

    while not CTRL_C:
        ret, frame = cap.read()

        # Binary segmentation
        # filtered = color_classification(frame)
        filtered = binary_classification(frame, bi_threshold)
        labels = image_representation(filtered)

        # Scene interpretation
        centers, huMoments, contours = feature_extraction(labels)

        # Draw contours
        # result = draw_contours(frame, centers, huMoments, contours)
        boxes = draw_boxes(frame, centers, huMoments, contours)

        cv.imshow('frame', np.hstack((boxes, cv.cvtColor(filtered, cv.COLOR_GRAY2BGR))))
        # cv.imshow('frame', np.hstack((boxes, filtered)))

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

        rate.sleep()
