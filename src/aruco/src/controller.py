#!/usr/bin/env python

import rospy
import sys
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist
from dynamic_reconfigure.server import Server
from aruco.cfg import PBVS_controllerConfig

ROTATION = 0.7
TRANSLATION = 0.15

def shutdownhook():
    global ctrl_c
    ctrl_c = True

def feedback_callback(feedback):
    pass

def param_update(config, level):
    global rot_speed, trans_speed, trans_target, rot_target, trans_tol, rot_tol, track_marker

    rot_speed = config["rot_speed"]
    trans_speed = config["trans_speed"]
    trans_target = config["trans_target"]
    rot_target = config["rot_target"]
    trans_tol = config["trans_tol"]
    rot_tol = config["rot_tol"]
    track_marker = config["track_marker"]

    print(config["track_marker"])

    return config

def update_command(marker_pose):
    global cmd
    print(rot_target + rot_tol)
    # Decide if the robot needs to turn
    if marker_pose.transform.translation.y > rot_target + rot_tol:
        cmd.angular.z = rot_speed # Turn Left
    elif marker_pose.transform.translation.y < rot_target - rot_tol:
        cmd.angular.z = -rot_speed # Turn Right
    else:
        cmd.angular.z = 0.0 # Stop

    # Decide if the robot needs to move forward/backward
    if marker_pose.transform.translation.x > trans_target + trans_tol:
        cmd.linear.x = trans_speed # Move forward
    elif marker_pose.transform.translation.x < trans_target - trans_tol:
        cmd.linear.x = -trans_speed # Move backward
    else:
        cmd.linear.x = 0.0 # Stop

def send_command():
    global cmd

    if cmd.linear.x == 0 and cmd.angular.z == 0:
        rospy.loginfo("Stop")
    elif cmd.linear.x == 0:
        rospy.loginfo("Turning")
    elif cmd.angular.z == 0:
        rospy.loginfo("Moving")
    else:
        rospy.loginfo("Moving and Turning")

    pub.publish(cmd)

if __name__ == '__main__':
    rospy.init_node('PBVS_controller')

    # Initialize tf2 variables
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # Initialize publishing vairables
    pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=1)
    cmd = Twist()

    # Initialize general variables
    ctrl_c = False
    rate = rospy.Rate(10)

    # These variables will be set/update by dynamic reconfigure
    rot_speed = 0.0
    trans_speed = 0.0
    trans_target = 0.0
    rot_target = 0.0
    trans_tol = 0.0
    rot_tol = 0.0
    track_marker = 0

    # Initialize dynamic reconfigure varibles
    srv = Server(PBVS_controllerConfig, param_update)

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        try:
            marker_pose = tfBuffer.lookup_transform("base_footprint", "%d_marker_frame"%track_marker, rospy.Time(), rospy.Duration(1.0))
            update_command(marker_pose)
            send_command()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
	    cmd.angular.z = 0.0
	    cmd.linear.x = 0.0
            send_command()

        rate.sleep()
