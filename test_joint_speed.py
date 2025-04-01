from utils.connect import RobotConnect
from utils.arm_mover import ArmMover
import time

# Set connection parameters
ip = "192.168.2.9"
port = 10000
credentials = ("jpotter2", "MSAScapstone")

# Create and open connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Instantiate ArmMover
mover = ArmMover(robot_connection)

# Example destination: using a sample destination pose
up = [0.53, -0.13, 0.92, 90, 0, 66]
down = [0.53, -0.13, 0.22, 90, 0, 66]
# Execute planning and movement with debug enabled
result = mover.move_to_pose(up)
result = mover.move_to_pose(down)
result = mover.move_to_pose(up)
result = mover.move_to_pose(down)

mover.set_default_cartesian_constraints(0.1) # Set joint speeds to 50% of max speed

result = mover.move_to_pose(up)
result = mover.move_to_pose(down)
result = mover.move_to_pose(up)
result = mover.move_to_pose(down)

robot_connection.close_connection()
