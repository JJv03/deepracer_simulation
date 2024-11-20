# deepRacer_simulator

## About the project

An open source library based on Gazebo and Ros to simulate DeepRacer. It has been developed on a Ubuntu 20.04 with Ros Noetic and Gazebo 11.11.0. 

## How to install and launch:

    git clone https://github.com/JJv03/deepracer_simulation
    rosdep install --from-paths src --ignore-src -r -y
    catkin build
    source ~/robot_ws/devel/setup.bash
    roslaunch deepracer_simulation racetrack_with_racecar.launch

[In case you have any errors](#possible-errors)
    
You can select between these three worlds: {easy_track | medium_track | hard_track}

## Useful commands:

Change speed:

    rostopic pub /<car_name>/<car_part>_wheel_velocity_controller/command std_msgs/Float64 "data: <value>" -r 10

Example use:

    rostopic pub /racecar/left_front_wheel_velocity_controller/command std_msgs/Float64 "data: 1.0" -r 10

Change wheel angle:

    rostopic pub /<car_name>/<car_side>_steering_hinge_position_controller/command std_msgs/Float64 "data: <value>" -r 10

Example use:

    rostopic pub /racecar/left_steering_hinge_position_controller/command std_msgs/Float64 "data: 0.0" -r 10

Use the camera:

    rqt_image_view

## Possible errors

In case you have this error:

    ERROR: cannot launch node of type [deepracer_simulation/servo_commands.py]: Cannot locate node of type [servo_commands.py] in package [deepracer_simulation]. Make sure file exists in package path and permission is set to executable (chmod +x)

Try to give execute rights to the file scripts/servo_commands.py (you can run this line from the launch directory to avoid changing directory, just for convenience):

    chmod +x ../scripts/servo_commands.py

In case you have this other error:

    /usr/bin/env: ‘python’: No such file or directory
    failed to start local process: /home/arob/robot_ws/src/deepracer_simulation/scripts/servo_commands.py /racecar/ackermann_cmd_mux/output:=/vesc/low_level/ackermann_cmd_mux/output __name:=servo_commands __log:=/home/arob/.ros/log/635cb86a-a78e-11ef-b994-536018386f45/servo_commands-7.log
    local launch of deepracer_simulation/servo_commands.py failed
    process[better_odom-8]: started with pid [4967]
    [servo_commands-7] process has died [pid -1, exit code 127, cmd /home/arob/robot_ws/src/deepracer_simulation/scripts/servo_commands.py /racecar/ackermann_cmd_mux/output:=/vesc/low_level/ackermann_cmd_mux/output __name:=servo_commands __log:=/home/arob/.ros/log/635cb86a-a78e-11ef-b994-536018386f45/servo_commands-7.log].
    log file: /home/arob/.ros/log/635cb86a-a78e-11ef-b994-536018386f45/servo_commands-7*.log

Try to install the following package(python-is-python3):

    sudo apt install python-is-python3
