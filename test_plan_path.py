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

# Execute planning and movement with debug enabled
for position in [mover.position1,mover.position2,mover.position3,mover.position4, mover.position5]:

    result = mover.plan_path_to_destination(position,debug=True)
    if result:
        
        print("Movement executed successfully.")
    else:
        print("Movement failed.")

    time.sleep(2)

#result = mover.plan_path_to_destination(object_dock,debug=True)
#if result:
    
#    print("Movement executed successfully.")
#else:
#    print("Movement failed.")

robot_connection.close_connection()
