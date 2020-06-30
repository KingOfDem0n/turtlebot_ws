#!/usr/bin/env python

import rospy
import sys
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist

def shutdownhook():
    global ctrl_c
    ctrl_c = True

def feedback_callback(feedback):
    pass

def update_command(marker_pose):
    global cmd
    # Decide if the robot needs to turn
    if marker_pose.transform.translation.y > 0.2:
        cmd.angular.z = 0.1 # Turn Left
    elif marker_pose.transform.translation.y < -0.2:
        cmd.angular.z = -0.1 # Turn Right
    else:
        cmd.angular.z = 0.0 # Stop

    # Decide if the robot needs to move forward/backward
    if marker_pose.transform.translation.x > 0.5:
        cmd.linear.x = 0.1 # Move forward
    elif marker_pose.transform.translation.x < 0.25:
        cmd.linear.x = -0.1 # Move backward
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

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    cmd = Twist()

    ctrl_c = False
    rate = rospy.Rate(10)

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        try:
            marker_pose = tfBuffer.lookup_transform("base_footprint", "406_marker_frame", rospy.Time())
            update_command(marker_pose)
            send_command()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

        rate.sleep()
