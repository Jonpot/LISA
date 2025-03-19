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

# Example destination: using home position as sample destination
destination = [0.305, 0.206, 0.42, 90, 0, 90]

# Execute planning and movement
result = mover.plan_path_to_destination(destination)
if result:
    print("Movement executed successfully.")
else:
    print("Movement failed.")

time.sleep(2)
robot_connection.close_connection()
