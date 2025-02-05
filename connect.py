"""
This file creates a connection to the arm and sets up configurations for the programs to work
Made for high-level control
"""
# You need these modules to connect to the robot
# Establish connection
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
# Messages to connect to the robot
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2
# Clients to accessed devices/services
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient

class RobotConnect:
    def __init__(self, ip, port, credentials):
        """
        :param ip: IP Address of the robot
        :param port: TCP port (default should be 10000)
        :param credentials: (login, password). Don't use admin/admin in your applications
        """
        # Hyperparameters
        self.ip = ip
        self.port = port
        self.credentials = credentials

        # Connection parameters created with a connection
        self.transport = None # TCP layer connection
        self.router = None # Router to arm's devices
        self.session_manager = None # Establish a session with uname/password
        self.device_manager = None # Information on connected devices
        self.base = None # Send commands and get feedback from the robot

        # Vision specific
        self.vision_config = None # Configurations specific to the vision object
        self.vision_device_id = None # Saving the id of the vision object saves time when getting its parameters

    def create_connection(self):
        """
        This function creates all required objects and connections to use the arm's services.
        It is easier to use the DeviceConnection utility class to create the router and then
        create the services you need (as done in the other examples).
        """
        # Set up API
        self.transport = TCPTransport() # TCP is to be used for high-level comms
        self.transport.connect(self.ip, self.port)  # TCP Connect is required for high-level servoing
        self.router = RouterClient(self.transport) # Router allows you to interact with devices

        # Create session
        # Anytime you want to use the robot, you need to create a session. This will use your credentials
        session_info = Session_pb2.CreateSessionInfo() # Message with the info to establish the session
        session_info.username = self.credentials[0]
        session_info.password = self.credentials[1]
        # Change this if you want, but there is usually no need
        session_info.session_inactivity_timeout = 60000   # (milliseconds)
        session_info.connection_inactivity_timeout = 2000 # (milliseconds)

        print("Creating session for communication")
        # Establish the connection with the credentials that were passed
        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(session_info)
        print("Session created")

        # Create required service interfaces
        self.device_manager = DeviceManagerClient(self.router)
        self.base = BaseClient(self.router)

        # Specifically save the vision device, as it is going to be used for the tracking
        self.vision_config = VisionConfigClient(self.router)

        # Identify the id of the vision device
        all_devices_info = self.device_manager.ReadAllDevices()
        vision_handles = [hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION]
        handle = vision_handles[0]
        self.vision_device_id = handle.device_identifier  # Save id to object properties

    def close_connection(self):
        """
        Call this when you're done with the arm
        """
        # Close API session
        self.session_manager.CloseSession()

        # Disconnect from the TCP object
        self.transport.disconnect()

        print("Session closed")

if __name__ == "__main__":
    # For debugging
    ip = "192.168.2.10" # TODO: Change to arm IP if not the same
    port = 10000 # Default TCP Port
    credentials = ("jpotter2", "MSAScapstone") # TODO: Change this in your application
    apiObject = RobotConnect(ip, port, credentials)
    apiObject.create_connection()
    apiObject.close_connection()