"""
This class commands the robot to go to some position and catch an object
"""
import cv2
import numpy as np

from utils.connect import RobotConnect
from kortex_api.autogen.messages import Base_pb2
import threading
import time
from scipy.spatial.transform import Rotation

from utils.vision import AprilTagDetector

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


        # Common Positions
        self.home = [0.57, 0.00, 0.42, 90, 0, 90] # position 3
        self.object_dock = [0.80, -0.04, 0.04, 90, 0, 90]
        self.sentry_position1 = [0.42, -0.40, 0.42, 90, 0, 46]
        self.sentry_position2 = [0.53, -0.23, 0.42, 90, 0, 66]
        self.sentry_position3 = self.home
        self.sentry_position4 = [0.54, 0.20, 0.42, 90, 0, 110]
        self.positions_list = [self.sentry_position1, self.sentry_position2, self.sentry_position3, self.sentry_position4]
        self.current_position = 2

    def robot_position(self) -> list[float]:
        """
        Returns the current position of the robot and euler angles
        :return: [x, y, z, theta_x, theta_y, theta_z]
        """
        current_pose = self.robot_connection.base.GetMeasuredCartesianPose()
        return [current_pose.x, current_pose.y, current_pose.z,
                current_pose.theta_x, current_pose.theta_y, current_pose.theta_z]

    def arbitrary_movement(self, x, y, z, theta_x, theta_y, theta_z, blocking = True) -> bool:
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

        #print(f"Moving to position ({x}, {y}, {z}) with rotation ({theta_x}, {theta_y}, {theta_z})")
        success = self._execute_movement(action, blocking)
        return success

    def move_to_pose(self, position: list[int], blocking = True) -> bool:
        """
        Move the arm to a pre-defined position
        :param position: list of 6 integers representing the position
        :return: if the operation was successful
        """
        return self.arbitrary_movement(position[0], position[1], position[2], position[3], position[4], position[5], blocking)

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

    def _execute_movement(self, action: Base_pb2.Action, blocking=True): # type: ignore
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
        if blocking:
            finished = e.wait(self.TIMEOUT_DURATION)
        else:
            finished = True
        self.robot_connection.base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished # Returns if the action was successful before the timeout

    def move_relative_to_tcp(self, movement_vector, scale: float=0.01, debug: bool = False):
        """
        Moves the arm based on a movement vector relative to the TCP (tool center point).
        
        :param movement_vector: A 3D numpy array [dx, dy, dz] in the user's frame.
        """
        # If robot is currently moving, pass
        if self.robot_connection.base.GetArmState().active_state == Base_pb2.ARMSTATE_SERVOING_PLAYING_SEQUENCE:
            return False

        # Get current TCP pose
        current_pose = self.robot_position()
        robot_position = np.array([current_pose[0], current_pose[1], current_pose[2]])
        robot_position = robot_position * 100 # multiply position by 100 (robot uses m, cm are more intuitive)
        
        euler_angles = np.array([current_pose[3]-90, current_pose[4], current_pose[5]-90])  # degrees
        if debug:
            print(f"Euclidian position: {robot_position}\nEuler angles: {euler_angles}")
        
        # Convert Euler angles to a rotation matrix
        rotation_matrix = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
        if debug:
            print(f"Rotation matrix: {rotation_matrix}")
        
        # Transform movement vector into the robot's frame
        transformed_movement = rotation_matrix @ movement_vector

        # Compute new target position
        new_position = robot_position + transformed_movement * scale
        new_position = new_position / 100 # Convert back to meters
        if debug:
            print(f"Robot position: {robot_position}\nTransformed movement: {transformed_movement*scale}\nNew position: {new_position}")

        # Move the arm
        if debug:
            print(f"Args: {new_position[0], new_position[1], new_position[2], current_pose.theta_x, current_pose.theta_y, current_pose.theta_z}")
        return self.arbitrary_movement(new_position[0], new_position[1], new_position[2],
                                       current_pose.theta_x, current_pose.theta_y, current_pose.theta_z, blocking = False)

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
        return self.move_gripper(0.23)

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
            prior_value = None
            same_count = 0
            while current_time - start < self.gripper_timeout:
                #print(f"Current value: {current_value}, target value: {value}")
                gripper_measure = self.robot_connection.base.GetMeasuredGripperMovement(gripper_request)
                current_value = gripper_measure.finger[0].value
                if current_value >= value:
                    print("Gripper closed")
                    # Stop the gripper
                    finger.value = 0
                    self.robot_connection.base.SendGripperCommand(gripper_command)
                    return True

                # If the gripper can't close anymore, stop it
                if prior_value == current_value:
                    same_count += 1
                    if same_count > 5:
                        print("Gripper can't close anymore")
                        finger.value = 0
                        self.robot_connection.base.SendGripperCommand(gripper_command)
                        return True # This is technically a success
                else:
                    same_count = 0
                prior_value = current_value

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


    def _move_to_current_position(self):
        """
        Move the robot to the current position in the list
        """
        self.move_to_pose(self.positions_list[self.current_position])

    def _move_to_next_position(self):
        """
        Move the robot to the next position in the list
        """
        self._move_to_current_position()
        self.current_position = (self.current_position + 1) % len(self.positions_list)
        self._move_to_current_position()

    def scan_for_apriltag(self, camera: AprilTagDetector, id: int) -> dict[str, int|float|np.ndarray]:
        """
        Scans for an apriltag with a specific ID
        :param camera: AprilTagDetector object
        :param id: ID of the apriltag
        :return: If the apriltag was found
        """
        frame_count = 0
        starting_pos = self.current_position

        # Try to start at the current position to avoid unnecessary movement
        self._move_to_current_position()
        detection = camera.detect_apriltag(id)
        if detection is not None:
            return detection

        self._move_to_next_position()
        while self.current_position != starting_pos:
            # Detect the apriltag
            detection = camera.detect_apriltag(id)
            if detection is not None:
                return detection

            frame_count += 1
            # Move in different positions if 10 frames have passed
            # We do this in case the camera was blurry or the apriltag was not yet in the field of view
            if frame_count >= 10:
                self._move_to_next_position()
                frame_count = 0

        print("Failed to find apriltag in scene.")
        return None

    def scan_for_unseen_apriltag(self,
                                  camera: AprilTagDetector,
                                  seen: set[int]) -> dict[str, int|float|np.ndarray]:
        """
        Similar to scan_for_apriltag, but searches for unseen apriltags. When
        a view has unseen apriltags, it will return the first of them.
        If all views have been exhausted, without finding any unseen apriltags,
        it will return None.

        :param camera: AprilTagDetector object
        :param seen: Set of seen apriltags
        :return: The detection if an unseen apriltag was found, or None if all views have been exhausted
        """
        frame_count = 0
        starting_pos = self.current_position

        # Try to start at the current position to avoid unnecessary movement
        self._move_to_current_position()
        detections = camera.detect_apriltags()
        for detection in detections:
            if detection['id'] not in seen:
                return detection

        self._move_to_next_position()
        while self.current_position != starting_pos:
            # Detect the apriltag
            detections = camera.detect_apriltags()
            for detection in detections:
                if detection['id'] not in seen:
                    return detection

            frame_count += 1
            # Move in different positions if 10 frames have passed
            # We do this in case the camera was blurry or the apriltag was not yet in the field of view
            if frame_count >= 10:
                self._move_to_next_position()
                frame_count = 0

        print("Failed to find unseen apriltag in scene.")
        return None



    def approach_apriltag_detection(self,
                                    camera: AprilTagDetector,
                                    detection: dict,
                                    threshold: float = 8,
                                    display_movement: bool = False,
                                    debug: bool = False) -> bool:
        """
        Approach the apriltag detection
        :param camera: AprilTagDetector object
        :param detection: Detection information
        :return: If the operation was successful
        """

        num_failed_attempts = 0
        x = ((-1 * detection['scale']) + 40) * 17 # 17 is a scaling factor determined experimentally. 40 represents the distance from the camera to the tcp
        y = detection['y'] * -1
        z = detection['z'] * -1
        while abs(x) > threshold:            
            if debug:
                print(f"Raw detection: {x, y, z}")
            self.move_relative_to_tcp([x, y, z]) # NOT blocking

            if display_movement:
                # Mover code isn't ready yet, just draw the circle on a canvas
                # convert to RGB
                image = cv2.cvtColor(detection['image'], cv2.COLOR_GRAY2BGR)
                image_size = image.shape
                y *= -1
                z *= -1
                y += image_size[1] / 2
                z += image_size[0] / 2
                cv2.circle(image, (int(y), int(z)), 10, (0, 255, 0))
                cv2.imshow("AprilTag Detection", image)

            new_detection = camera.detect_apriltag(detection['id'])
            if new_detection is not None:
                detection = new_detection

                y = detection['y'] * -1
                z = detection['z'] * -1
                x = ((-1 * detection['scale']) + 40) * 17 

                num_failed_attempts = 0
            else:
                num_failed_attempts += 1
                if num_failed_attempts >= 30:
                    # For 30 frames we haven't seen the apriltag. Likely out of view.
                    print("Lost apriltag during approach for 30 frames.")
                    return False
        return True

    def retrieve_apriltag_detection(self,
                                    camera: AprilTagDetector,
                                    detection: dict,
                                    display_movement: bool = False,
                                    debug: bool = False) -> bool:
        """
        Retrieve the apriltag
        :param camera: AprilTagDetector object
        :param detection: Detection information
        :return: If the operation was successful
        """
        # Move to the apriltag
        num_failed_attempts = 0
        while not self.approach_apriltag_detection(camera, detection, display_movement, debug):
            num_failed_attempts += 1
            self._move_to_current_position() # Back up and try again
            if num_failed_attempts >= 3:
                print("Failed to approach apriltag 3 times. Giving up.")
                return False # We tried 3 times and failed
            print("Failed to approach apriltag, reattempting")

        # Close the gripper
        self.close_gripper()

        # Move to the home position
        self._move_to_current_position() # back up directly first
        self.move_to_pose(self.home)
        self.move_to_pose(self.object_dock)
        self.open_gripper()
        self.move_to_pose(self.home)
        return True

    def find_and_retrieve_apriltag(self, camera: AprilTagDetector, id: int, display_movement: bool = False, debug: bool = False) -> bool:
        """
        Scans the scene for and retrieves the apriltag with a specific ID
        :param camera: AprilTagDetector object
        :param id: ID of the apriltag
        :return: If the operation was successful
        """
        detection = self.scan_for_apriltag(camera, id)
        if detection is None:
            return False

        print("Found apriltag in scene, attempting to retrieve.")
        return self.retrieve_apriltag_detection(camera, detection, display_movement, debug)