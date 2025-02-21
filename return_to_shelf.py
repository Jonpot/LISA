from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials)

robot.return_to_shelf()

# End the program
robot.exit()