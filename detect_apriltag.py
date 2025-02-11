from robotpy_apriltag import AprilTagDetector as apriltag
import cv2
import numpy as np
from utils.connect import RobotConnect
from utils.vision import BasicCamera
import math

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
#image = cv2.imread("image.png")
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

# Convert back to color 
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# Draw the detections
for detection in detections:
    point = detection.getCenter()
    print(point.x, point.y)
    x = int(point.x)
    y = int(point.y)
    corner = detection.getCorner(0)
    c_x = int(corner.x)
    c_y = int(corner.y)
    distance = math.sqrt((c_x - x)**2 + (c_y - y)**2)
    scale = int(distance / 3)
    cv2.circle(image, (x, y), scale, (0, 255, 0))

# Show the image
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)