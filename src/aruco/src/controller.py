#!/usr/bin/env python

import rospy
import sys
import tf2_ros
import numpy as np
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def shutdownhook():
    global ctrl_c
    ctrl_c = True

def feedback_callback(feedback):
    pass

def send_goal(marker_pose):
    goal = MoveBaseGoal()
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.header.frame_id = "map"
    
    goal.target_pose.pose.position.x = marker_pose.transform.translation.x - 0.5
    goal.target_pose.pose.position.y = marker_pose.transform.translation.y - 0.5
    goal.target_pose.pose.position.z = 0.0

    goal.target_pose.orientation.w = 1

    # For debugging purposes
    target_tf.header.stamp = rospy.Time.now()
    target_tf.header.frame_id = "base_footprint"
    target_tf.child_frame_id = "target"
    target_tf.transform.translation.x = goal.target_pose.pose.position.x
    target_tf.transform.translation.y = goal.target_pose.pose.position.y
    target_tf.transform.translation.z = goal.target_pose.pose.position.z
    target_tf.transform.rotation.w = goal.target_pose.orientation.w

    br.sendTransform(target_tf)

    client.send_goal(goal, feedback_cb=feedback_callback)
    # state_result = client.get_state()

if __name__ == '__main__':
    rospy.init_node('PBVS_controller')
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo('Waiting for action Server move_base')
    client.wait_for_server()
    rospy.loginfo('Action Server Found...move_base')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    br = tf2_ros.TransformBroadcaster()

    ctrl_c = False
    rate = rospy.Rate(1)

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        try:
            pose = tfBuffer.lookup_transform("base_footprint", "406_marker_frame", rospy.Time())
            send_goal(pose)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

        rate.sleep()
