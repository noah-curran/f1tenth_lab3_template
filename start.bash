#!/usr/bin/bash

cd /home/ntcurran/Code/f1tenth_labs/f1tenth_lab3_template/f1tenth_gym
pip3 install -e .

cd /home/ntcurran/Code/f1tenth_labs/f1tenth_lab3_template/f1tenth_gym_ros
mkdir -p sim_ws/src/f1tenth_gym_ros
cp -r config/ sim_ws/src/f1tenth_gym_ros/config 
rm -rf sim_ws/src/f1tenth_gym_ros/config/config

cp -r f1tenth_gym_ros/ sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros 
rm -rf sim_ws/src/f1tenth_gym_ros/f1tenth_gym_ros/f1tenth_gym_ros

cp -r launch/ sim_ws/src/f1tenth_gym_ros/launch 
rm -rf sim_ws/src/f1tenth_gym_ros/launch/launch

cp -r maps/ sim_ws/src/f1tenth_gym_ros/maps 
rm -rf sim_ws/src/f1tenth_gym_ros/maps/maps

cp -r resource/ sim_ws/src/f1tenth_gym_ros/resource 
rm -rf sim_ws/src/f1tenth_gym_ros/resource/resource

cp -r test/ sim_ws/src/f1tenth_gym_ros/test 
rm -rf sim_ws/src/f1tenth_gym_ros/test/test

cp -r package.xml sim_ws/src/f1tenth_gym_ros/package.xml 
cp -r setup.cfg sim_ws/src/f1tenth_gym_ros/setup.cfg 
cp -r setup.py sim_ws/src/f1tenth_gym_ros/setup.py 

source /opt/ros/foxy/setup.bash
cd /home/ntcurran/Code/f1tenth_labs/f1tenth_lab3_template/f1tenth_gym_ros/sim_ws/
sudo apt-get update --fix-missing
sudo rosdep install -i --from-path src --rosdistro foxy -y

declare -a pidArray

# pids+=($!)
rqt_graph &
pids+=($!)
ros2 run teleop_twist_keyboard teleop_twist_keyboard

for pid in "${pids[@]}"; do
    echo "Killing process with PID $pid"
    kill -SIGTERM $pid
    wait $pid  # Optionally wait for the process to terminate
done
