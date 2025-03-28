from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip, port, credentials, vi_mode={'speech': True, 'listening': True, 'reasoning': True, 'think_out_loud': False})

robot.move_object()

# End the program
robot.exit()