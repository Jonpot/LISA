# This script will use the camera to detect where an apriltag is in an image, and then move the robot to try to make the apriltag be
# in the center of the image (and 3 inches away from the camera).


import cv2
import numpy as np
from utils.connect import RobotConnect
from utils.vision import AprilTagDetector
from utils.arm_mover import ArmMover


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

home = [0.57, 0.00, 0.42, 90, 0, 90]
home_status = mover.arbitrary_movement(home[0], home[1], home[2], home[3], home[4], home[5])
if not home_status:
    print("Failed to move to home position")
    robot_connection.close_connection()
    exit()

import time
import math
while True:
    # Detect the apriltag
    detection_information = camera.detect_apriltag()
    if detection_information[0] is not None:
        y = detection_information[0] * -1
        z = detection_information[1] * -1
        #y = 0
        #z = 0
        x =  ((-1 *detection_information[2]) + 40) * 17
        image = detection_information[4]
        # Move the robot to center the apriltag
        print(f"Raw detection: {x, y, z}")  
        mover.move_relative_to_tcp([x, y, z])
        
        # if x, y, z very small, then we are at target, break!
        if abs(z) < 5:
            break

        # Mover code isn't ready yet, just draw the circle on a canvas
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_size = image.shape
        y *= -1
        z *= -1
        y += image_size[1] / 2
        z += image_size[0] / 2
        cv2.circle(image, (int(y), int(z)), 10, (0, 255, 0))
        cv2.imshow("AprilTag Detection", image)
    else:
        _, image = camera.video_capture.read()
        ## Show the image with a big red border to indicate no apriltag
        image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        cv2.imshow("AprilTag Detection", image)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# We are here, so close the gripper
mover.gentle_close_gripper()

object_dock = [0.80, -0.04, 0.04, 90, 0, 90]

# Move to object_dock position
home_status = mover.arbitrary_movement(home[0], home[1], home[2], home[3], home[4], home[5])
object_dock_status = mover.arbitrary_movement(object_dock[0], object_dock[1], object_dock[2], object_dock[3], object_dock[4], object_dock[5])

# Open gripper
open_status = mover.open_gripper()

# Move to home position
home_status = mover.arbitrary_movement(home[0], home[1], home[2], home[3], home[4], home[5])

# End the program
robot_connection.close_connection()