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

# Example destination (edit as needed to test edge of forbidden region or A* routing)
destination = [0.305, 0.206, 0.42, 90, 0, 90]

print("Planning path to destination...")
result = mover.plan_path_to_destination(destination)

if result:
    print("✅ Movement executed successfully.")
else:
    print("❌ Movement failed or was aborted.")

# Close connection
time.sleep(2)
robot_connection.close_connection()
