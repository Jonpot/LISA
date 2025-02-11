from robotpy_apriltag import AprilTagDetector as apriltag
import cv2
import numpy as np
from utils.connect import RobotConnect
from utils.vision import BasicCamera

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

# Start connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Attach the camera object to the same session
camera = BasicCamera(robot_connection)


# Take a picture
_, image = camera.video_capture.read()

# Show the image
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)

# Convert the image to grayscale (required for apriltag detection)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect the apriltag
detector = apriltag()

# Enable the tag family
detector.addFamily("tag25h9")
detections = detector.detect(image)
print(detections)

# Draw the detections
for detection in detections:
    for point in detection.corners:
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)

# Show the image
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)