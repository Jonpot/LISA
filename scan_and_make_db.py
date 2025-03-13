from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials, new_database=True)

robot.scan_and_populate_database_naive()

# End the program
robot.exit()