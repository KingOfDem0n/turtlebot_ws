#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np

from dynamic_reconfigure.server import Server
from feature_tracking.cfg import Feature_trackingConfig

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

def binary_classification(frame, threshold):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _ , thresh = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    return thresh

def image_representation(frame):
    global median_kernel
    blurred = cv.medianBlur(frame, int((median_kernel+1)*2+1))

    _, labels = cv.connectedComponents(blurred)
    labels = np.uint8(labels)

    _, contours, _ = cv.findContours(labels, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    moments = map(cv.moments, contours)
    moments_and_contours = zip(moments, contours)
    moments_and_contours = sorted(moments_and_contours, key=lambda x: x[0]["m00"], reverse=True)

    return np.array(moments_and_contours)

def calculate_center(moment):
    try:
        if moment["m00"] >= 0:
            center_x = int(round(moment["m10"]/moment["m00"]))
            center_y = int(round(moment["m01"]/moment["m00"]))
        else:
            center_x = -1
            center_y = -1
    except:
        center_x = 0
        center_y = 0
    return (center_x, center_y)

def draw_contours(frame, moments_and_contours, drawn_num=3):
    frame_copy = frame.copy()
    centers = map(calculate_center, moments_and_contours[:drawn_num, 0])
    phi = map(calculate_feature, moments_and_contours[:drawn_num, 0])

    # Draw contours and mark center dot
    result = cv.drawContours(frame_copy, moments_and_contours[:drawn_num, 1], -1, (0, 255, 0), 3)
    for center, p in zip(centers, phi):
        if center[0] != -1:
            result = cv.circle(result, center, 5, (255,0,0), -1)
            result = cv.putText(result, str(round(p[0], 2)), center, cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    return result

def calculate_feature(moment):
    # Calculate centeral momentum
    x, y = calculate_center(moment)
    u = np.zeros((4,4))
    u[1,1] = moment["m11"] - moment["m10"]*y
    u[2,0] = moment["m20"] - moment["m10"]*x
    u[0,2] = moment["m02"] - moment["m01"]*y
    u[2,1] = moment["m21"] - 2*x*moment["m11"] - y*moment["m20"] + 2*(x**2)*moment["m01"]
    u[1,2] = moment["m12"] - 2*y*moment["m11"] - x*moment["m02"] + 2*(y**2)*moment["m10"]
    u[3,0] = moment["m30"] - 3*x*moment["m20"] + 2*(x**2)*moment["m10"]
    u[0,3] = moment["m03"] - 3*y*moment["m02"] + 2*(y**2)*moment["m01"]

    # Calculate normalized moments
    n = np.zeros((4,4))
    for p,q in [[1,1],[2,0],[0,2],[2,1],[1,2],[3,0],[0,3]]:
        l = 1 + (p + q)/2
        n[p,q] = u[p,q]/(moment["m00"]**l)

    # Calculate features
    phi = np.zeros(7)
    phi[0] = n[2,0] + n[0,2]
    phi[1] = ((n[2,0] + n[0,2])**2) + 4*n[1,1]**2
    phi[2] = ((n[3,0]-3*n[1,2])**2) + (3*n[2,1]-n[0,3])**2
    phi[3] = ((n[3,0]+n[1,2])**2) +(n[2,1]+n[0,3])**2
    phi[4] = (n[3,0]-3*n[1,2])*(n[3,0]+n[1,2])*(((n[3,0]+n[1,2])**2)-3*(n[2,1]+n[0,3])**2) +\
             (3*n[2,1]-n[0,3])*(n[2,1]+n[0,3])*((3*(n[3,0]+n[1,2])**2)-(n[2,1]+n[0,3])**2)
    phi[5] = (n[2,0]-n[0,2])*(((n[3,0]+n[1,2])**2)-(n[2,1]+n[0,3])**2) +\
             4*n[1,1]*(n[3,0]+n[1,2])*(n[2,1]+n[0,3])
    phi[6] = (3*n[2,1]-n[0,3])*(n[3,0]+n[1,2])*(((n[3,0]+n[1,2])**2)-3*(n[2,1]+n[0,3])**2) +\
             (3*n[1,2]-n[3,0])*(n[2,1]+n[0,3])*((3*(n[3,0]+n[1,2])**2)-(n[2,1]+n[0,3])**2)

    return phi

def imshow_components(labels):
    # Ref: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    """
        Seizure warning! Because the label number changes order each frame, flashing bright color is possible
    """
    # Map component labels to hue val
    label_hue = np.uint8(255*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

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

        binary = binary_classification(frame, bi_threshold)

        moments_and_contours = image_representation(binary)

        calculate_feature(moments_and_contours[0][0])
        result = draw_contours(frame, moments_and_contours)

        cv.imshow('frame', result)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

        rate.sleep()
