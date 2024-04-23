source /opt/ros/foxy/setup.bash
cd /home/ntcurran/Code/f1tenth_labs/f1tenth_lab3_template/f1tenth_gym_ros/sim_ws
colcon build
source install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py