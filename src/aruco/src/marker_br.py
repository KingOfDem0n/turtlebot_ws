#!/usr/bin/env python

import rospy

import tf_conversions
import tf2_ros
import tf
import geometry_msgs.msg
from aruco_msgs.msg import MarkerArray

def marker_pose(msg):
	br = tf2_ros.TransformBroadcaster()
	t = geometry_msgs.msg.TransformStamped()
	t.header.frame_id = rospy.get_param('~frame_id',"camera_rgb_frame") # Need private parameters frame_id, or default to "camera_rgb_frame"

	for marker in msg.markers:
		t.header.stamp = marker.header.stamp
		t.child_frame_id = str(marker.id)+"_marker_frame"
		t.transform.translation.x = marker.pose.pose.position.x
		t.transform.translation.y = marker.pose.pose.position.y
		t.transform.translation.z = marker.pose.pose.position.z

		(r, p, y) = tf.transformations.euler_from_quaternion([marker.pose.pose.orientation.x,
								      marker.pose.pose.orientation.y,
								      marker.pose.pose.orientation.z,
								      marker.pose.pose.orientation.w])

		quaternion = tf.transformations.quaternion_from_euler(r+1.57079632679, p+3.14159265359, y)
		t.transform.rotation.x = quaternion[0]
		t.transform.rotation.y = quaternion[1]
		t.transform.rotation.z = quaternion[2]
		t.transform.rotation.w = quaternion[3]

		br.sendTransform(t)

if __name__ == '__main__':
	rospy.init_node('tf2_markers_broadcaster')
	rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, marker_pose)

	rospy.spin()
