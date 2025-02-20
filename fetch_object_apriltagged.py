# This script will use the camera to detect where an apriltag is in an image, and then move the robot to try to make the apriltag be
# in the center of the image (and 3 inches away from the camera).


import cv2
import numpy as np
from utils.connect import RobotConnect
from utils.vision import AprilTagDetector
from utils.arm_mover import ArmMover
import time
import math

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

# Start connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Attach the camera object to the same session
camera = AprilTagDetector(robot_connection)
mover = ArmMover(robot_connection)

home_status = mover.position_movement(mover.home)
if not home_status:
    print("Failed to move to home position")
    robot_connection.close_connection()
    exit()

mover.retrieve_apriltag(camera, id=2)

# End the program
robot_connection.close_connection()