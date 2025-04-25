from utils.robot import Robot
import time

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials, new_database=False, vi_mode={'speech': False, 'listening': False, 'reasoning': True, 'think_out_loud': False}, virtual_mode=False)

robot.scan_and_populate_database()


# End the program
robot.exit()