from utils.robot import Robot

# Parameters
ip = "192.168.2.9"
port = 10000 # Default TCP port
credentials = ("jpotter2", "MSAScapstone")

robot = Robot(ip,
              port,
              credentials,
              new_database=False,
              vi_mode={
                    "speech": True,
                    "listening": True,
                    "reasoning": True,
                    "think_out_loud": False
              },
              virtual_mode=True)

obj = robot._get_object_from_db(robot.vi.ask("What do you want me to bring you?"))

print(obj)