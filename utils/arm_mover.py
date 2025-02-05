"""
This class commands the robot to go to some position and catch an object
"""
import numpy as np

from utils.connect import RobotConnect
from kortex_api.autogen.messages import Base_pb2
import threading
import time

class ArmMover:
    def __init__(self, robot_connection: RobotConnect):
        # Link this object to an existing robot connection
        self.robot_connection = robot_connection

        # Movement parameters
        self.TIMEOUT_DURATION = 20 # Timeout for action
        self.gripper_timeout = 5 # Timeout for gripper actions

        # Position where the arm can see the table and track the target
        # This is hardcoded because this was the position that maximized precision in catching
        self.track_action = Base_pb2.Action()
        self.track_action.name = "Position to track the target"
        self.track_action.application_data = ""
        # Set the tracking pose
        cartesian_pose = self.track_action.reach_pose.target_pose # Pass by reference
        # Hardcoded
        cartesian_pose.x = 0.38 # (meters)
        cartesian_pose.y = 0.00  # (meters)
        cartesian_pose.z = 0.34  # (meters)
        cartesian_pose.theta_x = 180  # (degrees)
        cartesian_pose.theta_y = 0  # (degrees)
        cartesian_pose.theta_z = 90  # (degrees)

    def arbitrary_movement(self, x, y, z, theta_x, theta_y, theta_z) -> bool:
        """
        Move the arm to an arbitrary position
        :param x: x position in meters
        :param y: y position in meters
        :param z: z position in meters
        :param theta_x: x rotation in degrees
        :param theta_y: y rotation in degrees
        :param theta_z: z rotation in degrees
        :return: if the operation was successful
        """
        action = Base_pb2.Action()
        action.name = "Arbitrary movement"
        action.application_data = ""
        # Set the tracking pose
        cartesian_pose = action.reach_pose.target_pose
        # Hardcoded
        cartesian_pose.x = x
        cartesian_pose.y = y
        cartesian_pose.z = z
        cartesian_pose.theta_x = theta_x
        cartesian_pose.theta_y = theta_y
        cartesian_pose.theta_z = theta_z

        success = self._execute_movement(action)
        return success


    def _check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications
        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e=e):
            print("EVENT : " + \
                  Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
                    or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        return check

    def _execute_movement(self, action: Base_pb2.Action): # type: ignore
        """
        This is the function that actually executes the movement
        """
        # threading is necessary for us to check the status WHILE the robot goes to a position
        e = threading.Event()
        notification_handle = self.robot_connection.base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.robot_connection.base.ExecuteAction(action) # Send the desired action to the robot

        print("Waiting for movement to finish ...")
        finished = e.wait(self.TIMEOUT_DURATION)
        self.robot_connection.base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished # Returns if the action was successful before the timeout


    def object_tracking_position(self):
        """
        Move the arm to the pre-defined tracking position to maximize catching precision
        :return: if the operation was successful
        """
        print("Moving to tracking position ...")
        # The position has already been defined
        track_success = self._execute_movement(self.track_action)
        return track_success

    def catch_target(self, point: np.ndarray):
        """
        From the 3D point estimated by the vision module, move to the ball's position
        :param point: 3D numpy array
        :return: boolean if the operation was successful
        """
        # Open the gripper
        self.move_gripper(0)

        # Save the x and y position of the ball
        target_x = point[0]
        target_y = point[1]

        # Move to the object's vertical
        prime = Base_pb2.Action()
        prime.name = "Primed for target"
        prime.application_data = ""
        # Set the tracking pose
        cartesian_pose = prime.reach_pose.target_pose  # Pass by reference
        # Hardcoded
        cartesian_pose.x = target_x  # (meters)
        cartesian_pose.y = target_y  # (meters)
        cartesian_pose.z = 0.2  # (meters)
        cartesian_pose.theta_x = 180  # (degrees)
        cartesian_pose.theta_y = 0  # (degrees)
        cartesian_pose.theta_z = 90  # (degrees)

        success = self._execute_movement(prime)

        # If the arm has reached the target's vertical, go to target position
        if success:
            goto = Base_pb2.Action()
            goto.name = "Go to target"
            goto.application_data = ""
            # Set the tracking pose
            cartesian_pose = goto.reach_pose.target_pose  # Pass by reference
            # Hardcoded
            cartesian_pose.x = target_x  # (meters)
            cartesian_pose.y = target_y  # (meters)
            cartesian_pose.z = 0.02  # (meters)
            cartesian_pose.theta_x = 180  # (degrees)
            cartesian_pose.theta_y = 0  # (degrees)
            cartesian_pose.theta_z = 90  # (degrees)
            success = self._execute_movement(goto)

        # Grasp the ball
        if success:
            success = self.move_gripper(0.3)

        # Lift the ball
        if success:
            success = self._execute_movement(prime)

        # Open the gripper
        if success:
            self.move_gripper(0.05)

    def open_gripper(self):
        """
        Open the gripper
        :return: If operation was successful
        """
        return self.move_gripper(0.05)
    
    def close_gripper(self):
        """
        Close the gripper
        :return: If operation was successful
        """
        return self.move_gripper(0.9)
    
    def gentle_close_gripper(self):
        """
        Close the gripper gently
        :return: If operation was successful
        """
        return self.move_gripper(0.3)

    def move_gripper(self, value):
        """
        Open or close the gripper
        :param value: value in {0, 1}. 0 for open, 1 for closed.
        :return: If operation was successful
        """
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        # Set speed to open gripper
        print("Setting gripper position using velocity command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED


        # Create message that will allow us to get feedback on gripper status
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.robot_connection.base.GetMeasuredGripperMovement(gripper_request)
        current_value = gripper_measure.finger[0].value
        # Set velocity value depending on its sense (positive to open, negative to close. Yes, funny convention)
        # Close command
        if value > current_value:
            finger.value = -0.1
            self.robot_connection.base.SendGripperCommand(gripper_command)
            start = time.time()
            current_time = time.time()
            while current_time - start < self.gripper_timeout:
                print(f"Current value: {current_value}, target value: {value}")
                gripper_measure = self.robot_connection.base.GetMeasuredGripperMovement(gripper_request)
                current_value = gripper_measure.finger[0].value
                if current_value >= value:
                    print("Gripper closed")
                    # Stop the gripper
                    finger.value = 0
                    self.robot_connection.base.SendGripperCommand(gripper_command)
                    return True
                current_time = time.time()
            return False
        # Open command
        if value < current_value:
            finger.value = 0.1
            self.robot_connection.base.SendGripperCommand(gripper_command)
            start = time.time()
            current_time = time.time()
            while current_time - start < self.gripper_timeout:
                print(f"Current value: {current_value}, target value: {value}")
                gripper_measure = self.robot_connection.base.GetMeasuredGripperMovement(gripper_request)
                current_value = gripper_measure.finger[0].value
                if current_value <= value:
                    print("Gripper opened")
                    # Stop the gripper
                    finger.value = 0
                    self.robot_connection.base.SendGripperCommand(gripper_command)
                    return True
                current_time = time.time()
            return False


