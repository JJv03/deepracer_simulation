#!/bin/bash

# Update the package list
echo "Updating package list..."
sudo apt-get update

# Install necessary ROS Noetic packages for Gazebo
echo "Installing ROS Noetic packages for Gazebo..."
sudo apt-get install -y ros-noetic-gazebo-ros-pkgs
sudo apt-get install -y ros-noetic-gazebo-msgs
sudo apt-get install -y ros-noetic-gazebo-plugins
sudo apt-get install -y ros-noetic-gazebo-ros-control

# Install additional necessary packages
echo "Installing additional packages..."
sudo apt install -y ros-noetic-ackermann-msgs
sudo apt install -y python-is-python3
sudo apt install -y ros-noetic-effort-controllers

# Create workspace and compile
echo "Setting up ROS workspace..."
mkdir -p ~/robot_ws/src
cd ~/robot_ws
catkin_make

# Source the workspace environment
echo "Sourcing workspace environment..."
source ~/robot_ws/devel/setup.bash

# Display ROS package path
echo "ROS package path:"
echo $ROS_PACKAGE_PATH

# Create the deepracer_simulation package
echo "Creating deepracer_simulation package..."
cd ~/robot_ws/src
catkin_create_pkg deepracer_simulation gazebo_msgs gazebo_plugins gazebo_ros gazebo_ros_control mastering_ros_robot_description_pkg

# Deleting the deepracer_simulation directory files
rm -rf deepracer_simulation/*

echo "Setup completed."
