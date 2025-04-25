from utils.connect import RobotConnect
from utils.arm_mover import ArmMover

# Connect to real robot
ip = "192.168.2.9"
port = 10000
credentials = ("jpotter2", "MSAScapstone")
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Arm mover instance
mover = ArmMover(robot_connection)

# Define goal position
goal_position = mover.position1  # replace if needed

# Plan and visualize
mover.plan_path_to_destination(goal_position, debug=True)

robot_connection.close_connection()
