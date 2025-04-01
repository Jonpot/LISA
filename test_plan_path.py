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
destination = [0.305, 0.206, 0.42, 90, 0, 90]
object_dock = [-0.24, 0.45, 0.09, 90, 0, 171] # New position on second table
sentry_position1 = [0.42, -0.30, 0.42, 90, 0, 46]
sentry_position2 = [0.53, -0.13, 0.42, 90, 0, 66]
sentry_position4 = [0.54, 0.10, 0.22, 90, 0, 110]
sentry_position5 = [0.42, -0.30, 0.22, 90, 0, 46]
sentry_position6 = [0.53, -0.13, 0.22, 90, 0, 66]
sentry_position7 = [0.57, 0.00, 0.22, 90, 0, 90]
sentry_position8 = [0.54, 0.10, 0.22, 90, 0, 110]
# Execute planning and movement with debug enabled
result = mover.move_to_pose(destination)
if result:
    
    print("Movement executed successfully.")
else:
    print("Movement failed.")

time.sleep(2)

result = mover.move_to_pose(object_dock)
if result:
    
    print("Movement executed successfully.")
else:
    print("Movement failed.")

robot_connection.close_connection()
