from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials, new_database=False, vi_mode={'speech': False, 'listening': False, 'reasoning': False, 'think_out_loud': False})

#robot.scan_and_populate_database()

purple = robot._get_object_from_db(name="purple")

robot.move_object(object_to_move=purple, end_position=robot.database.positions["dock_1"])

# End the program
robot.exit()