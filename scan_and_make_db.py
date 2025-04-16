from utils.robot import Robot
import time

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials, new_database=True, vi_mode={'speech': False, 'listening': False, 'reasoning': True, 'think_out_loud': False})

robot.scan_and_populate_database()

vial = robot._get_object_from_db(name="vial")

robot.move_object(object_to_move=vial, end_position=robot.database.positions["dock_1"])

time.sleep(5)

robot.move_object(object_to_move=vial) # default is to return the object to its home

# End the program
robot.exit()