# This script will use the camera to detect where an apriltag is in an image, and then move the robot to try to make the apriltag be
# in the center of the image (and 3 inches away from the camera).

from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials)

robot.mover.find_and_move_apriltag(robot.camera, id=21, objective_pose=robot._get_next_empty_dock())

# End the program
robot.exit()