#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self._laser_scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self._cmd_drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # set PID gains
        self.kp = 5.
        self.ki = 0.03
        self.kd = 2.5

        # store history
        self.integral = 0.
        self.prev_error = 0.
        self.prev_time = self.get_clock().now()

        # store any necessary values you think you'll need
        self.desired_right_distance = 1.
        self.lookahead_distace = 0.1

    def get_range(self, range_data: LaserScan, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        if angle > range_data.angle_max or angle < range_data.angle_min:
            return float("inf")
        
        num_points = len(range_data.ranges)
        mid_point = num_points // 2
        angle_per_step = range_data.angle_increment
        angle_index = int(mid_point - angle / angle_per_step)

        self.get_logger().info(f"angle_increment ({np.rad2deg(range_data.angle_increment)}) angle_index ({mid_point + angle / angle_per_step})")

        return range_data.ranges[angle_index]

    def get_error(self, range_data: LaserScan, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        a_angle = np.deg2rad(30)
        b_angle = np.deg2rad(60)
        theta = b_angle - a_angle

        a = self.get_range(range_data, a_angle)
        b = self.get_range(range_data, b_angle)

        # Another method for getting D_t
        # phi = math.acos(a/b - math.cos(theta))
        # D_t = a * math.sin(phi)

        alpha = math.atan2( a * np.cos(theta) - b, a * np.sin(theta) )
        D_t = b * np.cos(alpha)
        D_t1 = D_t + self.lookahead_distace*np.sin(alpha)
        
        #  self.get_logger().info(f"D_t ({D_t}) a ({a}) b ({b}) phi ({0})")

        return dist - D_t1

    def pid_control(self, error):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error

        Returns:
            None
        """
        time = self.get_clock().now()
        delta_t = time - self.prev_time
        
        P = self.kp * error
        I = self.integral + self.ki * error * (delta_t.nanoseconds / 1e9)
        D = self.kd * (error - self.prev_error) / (delta_t.nanoseconds / 1e9)
        
        angle = P + I + D
        
        self.integral = I
        self.prev_error = error
        self.prev_time = time
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        if abs(angle) < np.deg2rad(10.):
            drive_msg.drive.speed = 1.5
        elif abs(angle) >= np.deg2rad(10.) and abs(angle) < np.deg2rad(20.):
            drive_msg.drive.speed = 1.
        else:
            drive_msg.drive.speed = 0.5
            
        #  self.get_logger().info(f"Publishing drive_msg: delta_t ({delta_t.nanoseconds / 1e9}) speed ({drive_msg.drive.speed}) steering_angle {np.rad2deg(drive_msg.drive.steering_angle)} w/ error ({error})")

        self._cmd_drive_pub.publish(drive_msg)

    def scan_callback(self, msg: LaserScan):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        error = self.get_error(msg, self.desired_right_distance)
        self.pid_control(error)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()