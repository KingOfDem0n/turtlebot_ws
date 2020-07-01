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

def callback(config, level):
    global rot_speed, trans_speed

    rot_speed = config["rot_speed"]
    trans_speed = config["trans_speed"]

    return config

def update_command(marker_pose):
    global cmd

    # Decide if the robot needs to turn
    if marker_pose.transform.translation.y > 0.1:
        cmd.angular.z = rot_speed # Turn Left
    elif marker_pose.transform.translation.y < -0.1:
        cmd.angular.z = -rot_speed # Turn Right
    else:
        cmd.angular.z = 0.0 # Stop

    # Decide if the robot needs to move forward/backward
    if marker_pose.transform.translation.x > 0.5:
        cmd.linear.x = trans_speed # Move forward
    elif marker_pose.transform.translation.x < 0.4:
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

    # Initialize dynamic reconfigure varibles
    srv = Server(PBVS_controllerConfig, callback)

    # Initialize general variables
    ctrl_c = False
    rate = rospy.Rate(10)
    rot_speed = 0.7
    trans_speed = 0.15

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        try:
            marker_pose = tfBuffer.lookup_transform("base_footprint", "406_marker_frame", rospy.Time(), rospy.Duration(1.0))
            update_command(marker_pose)
            send_command()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
	    cmd.angular.z = 0.0
	    cmd.linear.x = 0.0
            send_command()

        rate.sleep()
