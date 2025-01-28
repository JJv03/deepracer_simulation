# deepracer_simulation

## About the project

An open source library based on Gazebo and Ros to simulate DeepRacer. It has been developed on a Ubuntu 20.04 with Ros Noetic and Gazebo 11.11.0. 

## How to install and launch:

    git clone https://github.com/JJv03/deepracer_simulation
    rosdep install --from-paths src --ignore-src -r -y
    catkin build
    source ~/robot_ws/devel/setup.bash
    roslaunch deepracer_simulation racetrack_with_racecar.launch

## How to train (by default gui:=true):

    roslaunch deepracer_simulation train.launch (gui:=false)?

[In case you have any errors](#possible-errors)
    
You can select between these three worlds: {easy_track | medium_track | hard_track | 2022_april_open}

## Useful command:

Use the camera:

    rqt_image_view

## Possible errors

In case you have this error:

    ERROR: cannot launch node of type [deepracer_simulation/servo_commands.py]: Cannot locate node of type [servo_commands.py] in package [deepracer_simulation]. Make sure file exists in package path and permission is set to executable (chmod +x)
    ERROR: cannot launch node of type [deepracer_simulation/train.py]: Cannot locate node of type [train.py] in package [deepracer_simulation]. Make sure file exists in package path and permission is set to executable (chmod +x)
    
Try to give execute rights to the files scripts/servo_commands.py and train/train.py:

    chmod +x ../scripts/servo_commands.py
    chmod +x ../train/train.py
