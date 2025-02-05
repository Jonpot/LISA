from utils.connect import RobotConnect
from utils.arm_mover import ArmMover
import cv2
import keyboard


# Parameters
ip = "192.168.2.10"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

# Start connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Attach the movement object to the same session
mover = ArmMover(robot_connection)

# Move the arm to its object tracking position
#track_status = mover.object_tracking_position() # Blocking

# Move the arm arbitrarily
home = [0.57, 0.00, 0.42, 90, 0, 90]
object_dock = [0.80, 0.00, 0.10, 90, 0, 90]
jon = [0.32, 0.54, 0.27, 90, 0, 160]

# Move to home position
home_status = mover.arbitrary_movement(home[0], home[1], home[2], home[3], home[4], home[5])
if not home_status:
    print("Failed to move to home position")
    robot_connection.close_connection()
    exit()

# Move to object_dock position
object_dock_status = mover.arbitrary_movement(object_dock[0], object_dock[1], object_dock[2], object_dock[3], object_dock[4], object_dock[5])

# Close gripper
close_status = mover.gentle_close_gripper()

# Move to jon position
jon_status = mover.arbitrary_movement(jon[0], jon[1], jon[2], jon[3], jon[4], jon[5])

# Open gripper
open_status = mover.open_gripper()

# Move to home position
home_status = mover.arbitrary_movement(home[0], home[1], home[2], home[3], home[4], home[5])

# End the program
robot_connection.close_connection()