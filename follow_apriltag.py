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

import time
while True:
    # Detect the apriltag
    detection_information = camera.detect_apriltag()
    if detection_information[0] is not None:
        x = detection_information[0]
        y = detection_information[1]
        z = -1 * (detection_information[2] - 22)
        image = detection_information[4]
        # Move the robot to center the apriltag
        #mover.center_apriltag([x, y, z])
        
        # Mover code isn't ready yet, just draw the circle on a canvas
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_size = image.shape
        x += image_size[1] / 2
        y += image_size[0] / 2
        cv2.circle(image, (int(x), int(y)), int(z), (0, 255, 0))
        cv2.imshow("AprilTag Detection", image)
    else:
        _, image = camera.video_capture.read()
        # Show the image with a big red border to indicate no apriltag
        image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        cv2.imshow("AprilTag Detection", image)
    
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break