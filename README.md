# turtlebot_ws
This is a workshop on the turtlebot 2 in MRE lab. It is using kinetic distribution.

# Important
turtlebot packages must be installed first for this workshop to work. If the packages has not been install yet, simply run 

sudo apt install ros-kinetic-turtlebot

# To create a map
Start gmapping

roslaunch turtlebot_interface create_map.launch

then start a keyboard controller (or other controller of your choice)

roslaunch turtlebot_teleoperation keyboard.launch

# To navigate

roslaunch turtlebot_interface navigate.launch

# To run PBVS

roslaunch turtlebot_interface PBVS.launch debug:=false
