from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials, vi_mode={'speech': True, 'listening': True, 'reasoning': True, 'think_out_loud': False})

robot.return_to_shelf()

# End the program
robot.exit()