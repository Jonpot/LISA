from utils.connect import RobotConnect
from utils.arm_mover import ArmMover
import time
import random

# Set connection parameters
ip = "192.168.2.9"
port = 10000
credentials = ("jpotter2", "MSAScapstone")

# Create and open connection
robot_connection = RobotConnect(ip, port, credentials)
robot_connection.create_connection()

# Instantiate ArmMover
mover = ArmMover(robot_connection)

# List of sample destination poses (x, y, z, theta_x, theta_y, theta_z)
destinations = [
    [0.305, 0.206, 0.42, 90, 0, 90],
    [0.42, -0.30, 0.42, 90, 0, 46],
    [0.53, -0.13, 0.42, 90, 0, 66],
    [0.54, 0.10, 0.22, 90, 0, 110],
    [0.42, -0.30, 0.22, 90, 0, 46],
    [0.53, -0.13, 0.22, 90, 0, 66],
    [0.57, 0.00, 0.22, 90, 0, 90]
]

print("Starting random movement loop. Press Ctrl+C to exit.")
while True:
    dest = random.choice(destinations)
    print(f"\nMoving to destination: {dest}")
    # You can set debug=True if you want to simulate before executing.
    success = mover.plan_path_to_destination(dest, debug=True)
    if success:
        print("Movement executed successfully.")
    else:
        print("Movement failed.")
    time.sleep(2)

# Close connection when done (normally unreachable)
robot_connection.close_connection()
