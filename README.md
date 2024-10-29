# DeepRacer_Simulator

## About the project

An open source library based on Gazebo and Ros to simulate DeepRacer.

## How to install

Execute the setup_deepracer.sh:

    ./setup_deepracer.sh

Copy the deepracer_simulation files inside the ~/robot_ws/src/deepracer_simulation directory

To run the simulator, use the following commands:

    cd ~/robot_ws
    source ~/robot_ws/devel/setup.bash
    catkin_make
    cd src/deepracer_simulation/launch
    roslaunch deepracer_simulation racetrack_with_racecar.launch world_name:={easy_track | medium_track | hard_track}
