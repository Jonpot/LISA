"""
This class commands the robot to go to some position and catch an object
"""
import cv2
import numpy as np
import pybullet as p

from utils.connect import RobotConnect
from kortex_api.autogen.messages import Base_pb2
import threading
import time
from scipy.spatial.transform import Rotation

from utils.vision import AprilTagDetector
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        
        self.position1 = [0.59, 0.13, 0.16, 90, 0, 90]
        self.position2 = [0.59, -0.06, 0.16, 90, 0, 90]
        self.position3 = [0.59, -0.20, 0.16, 90, 0, 90]
        self.position4 = [0.59, -0.35, 0.16, 90, 0, 90]
        self.position5 = self.home
        self.position6 = [-0.37, 0.37, 0.05, -90, 180, 90]
        self.position7 = [-0.35, 0.65, 0.05, -90, 180, 90]
        
        
        self.positions_list = [self.position1, self.position2, self.position3, self.position4, self.position5, self.position6, self.position7]
        self.current_position = 0


        self.correction_forward_amount = 1000 # this should be fine-tuned experimentally
        self.correction_up_amount = 750

        self.cartesian_constraints = None

    def set_default_cartesian_constraints(self, max_speed_deg_s):
        """
        Set the default angular constraints for the robot. This is a workaround to set the speed and acceleration limits
        :param max_speed_deg_s: Maximum speed in degrees per second
        """
        constraints = Base_pb2.CartesianTrajectoryConstraint()
        constraints.speed.translation  = max_speed_deg_s

        self.cartesian_constraints = constraints


    @property
    def robot_position(self) -> list[float]:
        """
        Returns the current position of the robot and euler angles
        :return: [x, y, z, theta_x, theta_y, theta_z]
        """
        current_pose = self.robot_connection.base.GetMeasuredCartesianPose()
        return [current_pose.x, current_pose.y, current_pose.z,
                current_pose.theta_x, current_pose.theta_y, current_pose.theta_z]

    def arbitrary_cartesian_movement(self, x, y, z, theta_x, theta_y, theta_z, blocking = True) -> bool:
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

        # Attach constraints
        if self.cartesian_constraints is not None:
            action.reach_pose.constraint.CopyFrom(self.cartesian_constraints)
            print(f"Setting constraints to {self.cartesian_constraints.speed} deg/s")

        #print(f"Moving to position ({x}, {y}, {z}) with rotation ({theta_x}, {theta_y}, {theta_z})")
        success = self._execute_movement(action, blocking)
        return success

    def arbitrary_angular_movement(self, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, blocking = True) -> bool:
        """
        Move the arm to an arbitrary position
        :param theta_1: joint 1 angle in degrees
        :param theta_2: joint 2 angle in degrees
        :param theta_3: joint 3 angle in degrees
        :param theta_4: joint 4 angle in degrees
        :param theta_5: joint 5 angle in degrees
        :param theta_6: joint 6 angle in degrees
        :return: if the operation was successful
        """
        action = Base_pb2.Action()
        action.name = "Arbitrary movement"
        action.application_data = ""
        # Set the tracking pose
        joint_angles = action.reach_joint_angles.joint_angles
        
        actuator_count = self.robot_connection.base.GetActuatorCount()
        angles = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        #print(f"Moving to position ({theta_1}, {theta_2}, {theta_3}, {theta_4}, {theta_5}, {theta_6})")
        success = self._execute_movement(action, blocking)
        return success
    
    def move_to_pose(self, position: list[int], blocking=True, interval=0.5, duration=10, logging: bool = False) -> bool:
        """
        Move the arm to a pre-defined position
        :param position: list of 6 integers representing the position
        :param blocking: if the operation should block until completion
        :param interval: time interval in seconds to log the positions
        :param duration: total duration in seconds to log the positions
        :return: if the operation was successful
        """
        if logging:
            log_entries = []

            def log_positions():
                start_time = time.time()
                last_log_time = start_time
                while time.time() - start_time < duration:
                    current_time = time.time()
                    if current_time - last_log_time >= interval:
                        # Get the current timestamp with milliseconds
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                        # Append the dynamically measured robot position and timestamp to the log entries list
                        log_entries.append([timestamp] + self.robot_position)

                        # Update the last log time
                        last_log_time = current_time

            # Start the logging thread
            logging_thread = threading.Thread(target=log_positions)
            logging_thread.start()

        # Start the movement
        #success = self.arbitrary_cartesian_movement(position[0], position[1], position[2], position[3], position[4], position[5], blocking)

        success = self.plan_path_to_destination(position, blocking=blocking)

        if logging:
            # Wait for the logging thread to finish
            logging_thread.join()

            # Write all log entries to a CSV file after the movement is complete
            with open('positions_log.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(log_entries)

        return success

    def move_to_object_dock(self, blocking = True) -> bool:
        """
        Move the arm to the object dock position. Because the object dock is sometimes complicated to get to, this
        uses angular movement to get to the object dock position.
        :return: if the operation was successful
        """
        self.move_to_pose(self.home, blocking)
        #self.arbitrary_angular_movement(225, 12, 230, 0, 55, 90)
        return self.move_to_pose(self.object_dock, blocking)
        

    def move_home_from_dock(self, blocking = True) -> bool:
        """
        Move the arm from the object dock to the home position
        :return: if the operation was successful
        """
        #self.arbitrary_angular_movement(225, 12, 230, 0, 55, 90)
        #self.arbitrary_angular_movement(0, 15, 230, 0, 55, 90)
        return self.move_to_pose(self.home, blocking)

    def _check_for_end_or_abort(self, e, debug=False):
        """Return a closure checking for END or ABORT notifications
        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e=e):
            if debug:
                print("EVENT : " + \
                      Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
                    or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        return check

    def _execute_movement(self, action: Base_pb2.Action, blocking=True, debug=False): # type: ignore
        """
        This is the function that actually executes the movement
        """
        # threading is necessary for us to check the status WHILE the robot goes to a position
        e = threading.Event()
        notification_handle = self.robot_connection.base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        if debug:
            print("Executing action")
        self.robot_connection.base.ExecuteAction(action) # Send the desired action to the robot

        if debug:
            print("Waiting for movement to finish ...")
        if blocking:
            finished = e.wait(self.TIMEOUT_DURATION)
        else:
            finished = True
        self.robot_connection.base.Unsubscribe(notification_handle)

        if finished:
            if debug:
                print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished # Returns if the action was successful before the timeout

    def move_relative_to_tcp(self, movement_vector, scale: float=0.01, blocking: bool = False, debug: bool = False):
        """
        Moves the arm based on a movement vector relative to the TCP (tool center point).
        
        :param movement_vector: A 3D numpy array [dx, dy, dz] in the user's frame.
        """
        # If robot is currently moving, pass
        if self.robot_connection.base.GetArmState().active_state == Base_pb2.ARMSTATE_SERVOING_PLAYING_SEQUENCE:
            return False

        # Get current TCP pose
        current_pose = self.robot_position
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
            print(f"Args: {new_position[0], new_position[1], new_position[2], current_pose[3], current_pose[4], current_pose[5]}")

        return self.arbitrary_cartesian_movement(new_position[0], new_position[1], new_position[2],
                                       current_pose[3], current_pose[4], current_pose[5], blocking = blocking)

    def open_gripper(self):
        """
        Open the gripper
        :return: If operation was successful
        """
        return self.move_gripper(0.008)
    
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

    def waggle_gripper(self):
        """
        Waggle the gripper
        :return: If operation was successful
        """
        self.move_gripper(0.009)
        self.move_gripper(0.035)
        self.move_gripper(0.009)
        self.move_gripper(0.035)
        self.move_gripper(0.009)
        self.move_gripper(0.035)
        return True

    def move_gripper(self, value, debug=False):
        """
        Open or close the gripper
        :param value: value in {0, 1}. 0 for open, 1 for closed.
        :return: If operation was successful
        """
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        # Set speed to open gripper
        if debug:
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
                if debug:
                    print(f"Current value: {current_value}, target value: {value}")
                gripper_measure = self.robot_connection.base.GetMeasuredGripperMovement(gripper_request)
                current_value = gripper_measure.finger[0].value
                if current_value >= value:
                    if debug:
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
                if debug:
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
        print(f"Moving to position {self.current_position}")
        self.move_to_pose(self.positions_list[self.current_position])

    def _move_to_next_position(self):
        """
        Move the robot to the next position in the list
        """
        #self.move_to_pose(self.home)
        self.current_position = (self.current_position + 1) % len(self.positions_list)
        self._move_to_current_position()

    def scan_for_apriltag(self, camera: AprilTagDetector, id: int, debug: bool = False) -> dict[str, int|float|np.ndarray]:
        """
        Scans for an apriltag with a specific ID
        :param camera: AprilTagDetector object
        :param id: ID of the apriltag
        :return: If the apriltag was found
        """
        starting_pos = self.current_position

        # Try to start at the current position to avoid unnecessary movement
        self._move_to_current_position()
        time.sleep(0.5)
        detection = camera.detect_apriltag(id, debug=debug)
        if detection is not None:
            return detection

        self._move_to_next_position()
        time.sleep(0.5)
        while self.current_position != starting_pos:
            # Detect the apriltag
            detection = camera.detect_apriltag(id, debug=debug)
            if detection is not None:
                return detection

            self._move_to_next_position()
            time.sleep(0.5)

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

            self._move_to_next_position()

        print("Failed to find unseen apriltag in scene.")
        return None



    def approach_apriltag_detection(self,
                                    camera: AprilTagDetector,
                                    detection: dict,
                                    threshold: float = 300,
                                    debug: bool = False) -> bool:
        """
        Approach the apriltag detection
        :param camera: AprilTagDetector object
        :param detection: Detection information
        :return: If the operation was successful
        """

        num_failed_attempts = 0
        # Note that what we would call "xyz" the camera calls "yzx"
        x = ((-1 * detection['z']) + 40) * 17 # 17 is a scaling factor determined experimentally. 40 represents the distance from the camera to the tcp
        y = detection['x'] * -1
        z = (detection['y'] * -1)
        while abs(x) > threshold: 
            if True:
                print(f"Approaching apriltag, x: {x}, y: {y}, z: {z}, threshold: {abs(x)} is greater than {threshold}")  
                print(f"Raw detection: {x, y, z}")

            # only approach if y and z are relatively cenetered
            if abs(y) > 50 or abs(z) > 50:
                if debug:
                    print(f'Y or Z is too far from center, not approaching. y: {y}, z: {z}')
                self.move_relative_to_tcp([0, y, z])
            else:
                self.move_relative_to_tcp([x, y, z]) # NOT blocking

            new_detection = camera.detect_apriltag(detection['id'])
            if new_detection is not None:
                detection = new_detection

                y = detection['x'] * -1
                z = detection['y'] * -1  + 60 # 10 to make it a little higher for most apriltags
                x = ((-1 * detection['z']) + 40) * 17 

                num_failed_attempts = 0
            else:
                num_failed_attempts += 1
                if num_failed_attempts >= 30:
                    # For 30 frames we haven't seen the apriltag. Likely out of view.
                    if debug:
                        print("Lost apriltag during approach for 30 frames.")
                    return False
        if debug:
            print(f"Approaching apriltag, x: {x}, y: {y}, z: {z}, threshold: {abs(x)} is less than(?) {threshold}")  
        return True

    def move_apriltag_detection(self,
                                camera: AprilTagDetector,
                                detection: dict,
                                objective_pose: list[float],
                                debug: bool = False) -> bool:
        """
        Retrieve the apriltag
        :param camera: AprilTagDetector object
        :param detection: Detection information
        :return: If the operation was successful
        """
        objective_pose = objective_pose.copy() # Copy the list to avoid modifying the original one
        # Move to the apriltag
        num_failed_attempts = 0
        while not self.approach_apriltag_detection(camera, detection, debug=debug):
            num_failed_attempts += 1
            self._move_to_current_position() # Back up and try again
            if num_failed_attempts >= 3:
                print("Failed to approach apriltag 3 times. Giving up.")
                return False # We tried 3 times and failed
            print("Failed to approach apriltag, reattempting")

        # Close the gripper
        self.close_gripper()

        # Move to the home position
        self.move_relative_to_tcp([0, 0, self.correction_up_amount], blocking=True)
        self.move_relative_to_tcp([-self.correction_forward_amount, 0, 0], blocking=True)
        self._move_to_current_position() # back up directly first

        print(f"Objective pose before adding correction: {objective_pose}")
        objective_pose[2] += self.correction_up_amount/10000 # add the correction amount to the z position
        print(f"Objective pose after adding correction: {objective_pose}")
        self.move_to_pose(objective_pose)
        
        # Move forward ~10cm to ensure the object is placed on the dock
        self.move_relative_to_tcp([self.correction_forward_amount, 0, 0], blocking=True)
        self.move_relative_to_tcp([0, 0, -self.correction_up_amount], blocking=True)

        self.open_gripper()
        self.move_relative_to_tcp([0, 0, self.correction_up_amount], blocking=True)
        self.move_relative_to_tcp([-self.correction_forward_amount, 0, 0], blocking=True)

        return True

    def find_and_move_apriltag(self, camera: AprilTagDetector, id: int, objective_pose: list[float], debug: bool = True) -> bool:
        """
        Scans the scene for and retrieves the apriltag with a specific ID
        :param camera: AprilTagDetector object
        :param id: ID of the apriltag
        :return: If the operation was successful
        """
        detection = self.scan_for_apriltag(camera, id, debug=debug)
        if detection is None:
            return False

        print("Found apriltag in scene, attempting to retrieve.")
        return self.move_apriltag_detection(camera, detection, debug=debug, objective_pose=objective_pose)

    def calculate_forbidden_ellipse(self) -> tuple[list[float], float, float, float, float]:
        """
        Returns the parameters of the forbidden ellipse.
        PLEASE UPDATE these parameters according to your workspace measurements.
        For example:
          - center_xy: set to the actual center of the dangerous region.
          - a and b: set to the radii that actually block unreachable regions.
          - z_center and delta_z: set to the vertical limits of the forbidden zone.
        """
        center_xy = [0.03, 0.0]  # Update if your unsafe region is offset.
        a = 0.46             # Adjust these radii to cover only the truly unsafe area.
        b = 0.30
        z_center = 0.34       # Adjust based on your safe z.
        delta_z = 0.18        # Lower delta_z to shrink the forbidden vertical zone.
        return center_xy, a, b, z_center - delta_z, z_center + delta_z

    def is_path_safe_ellipse(self, start: list[float], end: list[float]) -> bool:
        """
        Checks whether the linear path from start to end avoids the forbidden ellipse.
        Samples N points and uses a margin factor increased by an extra_margin plus an extra tolerance for x,y and z.
        """
        center_xy, a, b, z_min, z_max = self.calculate_forbidden_ellipse()
        z_extra = 0.01  # additional tolerance in z
        xy_extra = 0.01  # additional tolerance in x and y
        N = 40  # number of samples
        base_margin = 0.5  
        # extra_margin = 0.25  # additional delta for XY
        margin = base_margin + xy_extra
        start_np = np.array(start[:3])
        end_np = np.array(end[:3])
        for t in np.linspace(0, 1, N):
            sample = start_np + t * (end_np - start_np)
            x, y, z = sample
            ellipse_val = ((x - center_xy[0]) / a)**2 + ((y - center_xy[1]) / b)**2
            if ellipse_val < margin and (z_min - z_extra <= z <= z_max + z_extra):
                return False
        return True

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    def plan_path_to_destination(self, dest: list[float], debug: bool = False, blocking: bool = True) -> bool:
        """
        Plans a safe path from current position to destination avoiding a forbidden ellipse region.
        If the direct path is unsafe, applies a buffer in +y direction and projects waypoints when needed.
        All waypoints receive a +y offset to improve clearance.
        """
        current_pose = self.robot_position
        forbidden_center, a, b, _, _ = self.calculate_forbidden_ellipse()
        threshold = 0.275

        def is_safe_line(p1, p2, center, a, b, margin=0.07):
            steps = 20
            for alpha in np.linspace(0, 1, steps):
                pt = p1 * (1 - alpha) + p2 * alpha
                dx = (pt[0] - center[0]) / (a + margin)
                dy = (pt[1] - center[1]) / (b + margin)
                if dx**2 + dy**2 < 1:
                    return False
            return True

        def project_and_offset_outside(pt, center, a, b, offset):
            v = pt - center
            if np.allclose(v, 0):
                v = np.array([1e-3, 0])
            scale = np.sqrt((v[0] / a)**2 + (v[1] / b)**2)
            boundary = center + v / scale
            direction = (boundary - center) / np.linalg.norm(boundary - center)
            return boundary + direction * offset

        start = np.array(current_pose[:2])
        goal = np.array(dest[:2])
        z0, z1 = current_pose[2], dest[2]

        if is_safe_line(start, goal, forbidden_center, a, b):
            if debug:
                print("Direct path is safe.")
            return self.arbitrary_cartesian_movement(*dest, blocking=blocking)

        if debug:
            print("Direct path is unsafe.")

        # Step 1: Move current position +0.10 in Y
        buffered_pose = current_pose.copy()
        #buffered_pose[1] += 0.2
        if not self.arbitrary_cartesian_movement(*buffered_pose, blocking=True):
            print("Failed to move to initial buffered position.")
            return False

        # Step 2: Start path planning from new pose
        current_pose = self.robot_position
        start = np.array(current_pose[:2])
        rpy_dest = dest[3:6]
        waypoints = [start]

        if not is_safe_line(start, goal, forbidden_center, a, b):
            mid = (start + goal) / 2
            safe_mid = project_and_offset_outside(mid, forbidden_center, a, b, threshold)

            if not is_safe_line(start, safe_mid, forbidden_center, a, b):
                mid1 = (start + safe_mid) / 2
                safe_mid1 = project_and_offset_outside(mid1, forbidden_center, a, b, threshold)
                waypoints.append(safe_mid1)

            waypoints.append(safe_mid)

            if not is_safe_line(safe_mid, goal, forbidden_center, a, b):
                mid2 = (safe_mid + goal) / 2
                safe_mid2 = project_and_offset_outside(mid2, forbidden_center, a, b, threshold)
                waypoints.append(safe_mid2)

        waypoints.append(goal)

        # Apply +0.05 Y-buffer to all waypoints
        buffered_waypoints = []
        n = len(waypoints) - 1
        for i, (x, y) in enumerate(waypoints):
            alpha = i / (len(waypoints) - 1)
            z = z0 * (1 - alpha) + z1 * alpha
            
            #buffered_waypoints.append([x, y , z] + list(rpy))

            # at every point, the rpy should be the angle facing directly away from the origin
            rpy = [90, 0, ((np.arctan2(y - forbidden_center[1], x - forbidden_center[0]) * 180 / np.pi) + 90) % 360]
            
            buffered_waypoints.append([x, y + 0.05, z] + list(rpy))

        # set the start/destination rpi to the one we want
        #buffered_waypoints[0][3:6] = rpy_start
        buffered_waypoints[-1][3:6] = rpy_dest

        if debug:
            fig, ax = plt.subplots()
            wp_np = np.array(buffered_waypoints)
            ax.plot(wp_np[:, 0], wp_np[:, 1], 'y-', label='Planned Path')
            ax.scatter(start[0], start[1], c='g', label='Start', marker='x')
            ax.scatter(goal[0], goal[1] + 0, c='b', label='Goal', marker='x')
            for i, pt in enumerate(buffered_waypoints[1:-1], 1):
                ax.scatter(pt[0], pt[1] + 0, marker='x', label=f'Waypoint {i}, angle {pt[3:6]}', alpha=0.8)

            theta = np.linspace(0, 2 * np.pi, 100)
            xe = forbidden_center[0] + a * np.cos(theta)
            ye = forbidden_center[1] + b * np.sin(theta)
            ax.plot(xe, ye, 'r--', label='Forbidden Zone')

            ax.set_aspect('equal')
            ax.legend()
            ax.set_title("Path with +Y Buffers & Forbidden Zone Avoidance")
            plt.show()

        print(f"Buffered waypoints: {buffered_waypoints}")
        return self.execute_path(buffered_waypoints, delay=0.01, steps_per_segment=3)




    def arc_around_obstacle(self, start, goal, center, height_offset=0.05, num_points=50):
        """
        Plan a clean fixed-height arc *above* the forbidden ellipse in XY.
        The arc radius is set based on start/goal distance to center + margin.
        """
        import numpy as np
        start = np.array(start[:2])
        goal = np.array(goal[:2])
        center = np.array(center[:2])

        # Define radius slightly larger than forbidden ellipse
        r_start = np.linalg.norm(start - center)
        r_goal = np.linalg.norm(goal - center)
        radius = max(r_start, r_goal) + 0.03  # fixed clearance

        # Angle from center to start/goal
        theta_start = np.arctan2(start[1] - center[1], start[0] - center[0])
        theta_goal = np.arctan2(goal[1] - center[1], goal[0] - center[0])

        # Shortest arc direction
        if theta_start < 0: theta_start += 2 * np.pi
        if theta_goal < 0: theta_goal += 2 * np.pi
        delta = theta_goal - theta_start
        if delta > np.pi:
            theta_goal -= 2 * np.pi
        elif delta < -np.pi:
            theta_goal += 2 * np.pi

        thetas = np.linspace(theta_start, theta_goal, num_points)
        arc_xy = center.reshape(1, 2) + radius * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)

        # Apply a fixed height boost (Z) or hold Z constant
        arc_2d = arc_xy.tolist()
        return arc_2d


    def interpolate_path(self, waypoints, steps_per_segment=5):
        """
        Linearly interpolate between each pair of waypoints to generate smoother substeps.
        """
        interpolated = []
        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i])
            end = np.array(waypoints[i + 1])
            for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
                point = (1 - t) * start + t * end
                interpolated.append(point.tolist())
        interpolated.append(waypoints[-1])  # include final point
        return interpolated

    def execute_path(self, waypoints: list[list[float]], steps_per_segment: int = 5, delay: float = 0.01) -> bool:
        """
        Executes the interpolated path.
        Intermediate waypoints are moved non-blockingly, and the final move is blocking to ensure the destination is reached.
        """
        #interpolated_path = self.interpolate_path(waypoints, steps_per_segment)
        interpolated_path = waypoints  # Use the original waypoints for now
        for point in interpolated_path[:-1]:
            self.arbitrary_cartesian_movement(*point, blocking=True)  # ← non-blocking for smooth motion
            if delay > 0:
                time.sleep(delay)
        
        # Final move is blocking
        return self.arbitrary_cartesian_movement(*interpolated_path[-1], blocking=True)

    def smooth_path(self, path_xyz: list[tuple]) -> list[tuple]:
        """
        Smooths the given path using a simple moving average filter.
        The first and last point remain unchanged.
        """
        if len(path_xyz) < 3:
            return path_xyz
        smoothed = [path_xyz[0]]
        for i in range(1, len(path_xyz) - 1):
            prev_pt = path_xyz[i - 1]
            cur_pt = path_xyz[i]
            next_pt = path_xyz[i + 1]
            avg_pt = (round((prev_pt[0] + cur_pt[0] + next_pt[0]) / 3, 3),
                      round((prev_pt[1] + cur_pt[1] + next_pt[1]) / 3, 3),
                      round((prev_pt[2] + cur_pt[2] + next_pt[2]) / 3, 3))
            smoothed.append(avg_pt)
        smoothed.append(path_xyz[-1])
        return smoothed

    def visualize_path_and_forbidden_zone(self, path_xyz: list[tuple], forbidden_params: tuple):
        """
        Visualizes the forbidden zone and the planned path in 3D.
        :param path_xyz: List of (x, y, z) tuples representing the path.
        :param forbidden_params: Tuple containing forbidden ellipse parameters (center_xy, a, b, z_min, z_max).
        """
        center_xy, a, b, z_min, z_max = forbidden_params

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the forbidden zone as an ellipsoid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = a * np.outer(np.cos(u), np.sin(v)) + center_xy[0]
        y = b * np.outer(np.sin(u), np.sin(v)) + center_xy[1]
        z = (z_max - z_min) * np.outer(np.ones(np.size(u)), np.cos(v)) + z_min
        ax.plot_surface(x, y, z, color='r', alpha=0.3, label="Forbidden Zone")

        # Plot the planned path
        path_xyz = np.array(path_xyz)
        ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], color='b', label="Planned Path")

        # Highlight points inside the forbidden zone
        for point in path_xyz:
            x, y, z = point
            ellipse_val = ((x - center_xy[0]) / a)**2 + ((y - center_xy[1]) / b)**2
            if ellipse_val < 1.0 and z_min <= z <= z_max:
                ax.scatter(x, y, z, color='orange', label="Inside Forbidden Zone")

        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def a_star_path_planning(self, start: list[float], goal: list[float], resolution=0.02) -> list[list[float]]:
        """
        A* path planning in 3D (x, y, z) over a grid. Orientation is linearly interpolated.
        The safety check (is_safe) is applied on the (x,y,z) coordinates.
        """
        from queue import PriorityQueue

        center_xy, a, b, z_min, z_max = self.calculate_forbidden_ellipse()

        def is_safe(x, y, z):
            # Add stricter margin to forbidden zone
            margin_factor = 1.2  # Increase margin to ensure safety
            val = ((x - center_xy[0]) / (a * margin_factor))**2 + ((y - center_xy[1]) / (b * margin_factor))**2
            if val < 1.0 and (z_min <= z <= z_max):
                return False
            return True

        def heuristic(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        sx, sy, sz = start[:3]
        gx, gy, gz = goal[:3]
        start_state = (sx, sy, sz)
        goal_state = (gx, gy, gz)

        visited = set()
        queue = PriorityQueue()
        queue.put((0, start_state))
        parent = {}

        while not queue.empty():
            cost, current = queue.get()
            if current in visited:
                continue
            visited.add(current)

            if heuristic(current, goal_state) < resolution * 1.5:
                break

            for dx in [-resolution, 0, resolution]:
                for dy in [-resolution, 0, resolution]:
                    for dz in [-resolution, 0, resolution]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (round(current[0] + dx, 3),
                                    round(current[1] + dy, 3),
                                    round(current[2] + dz, 3))
                        if neighbor in visited:
                            continue
                        if not is_safe(*neighbor):
                            continue
                        new_cost = heuristic(current, neighbor)
                        total_cost = cost + new_cost + heuristic(neighbor, goal_state)
                        queue.put((total_cost, neighbor))
                        parent[neighbor] = current

        # Reconstruct path in 3D.
        path_xyz = []
        if parent:
            current = min(parent, key=lambda p: heuristic(p, goal_state))
        else:
            current = start_state
        while current in parent:
            path_xyz.append(current)
            current = parent[current]
        path_xyz.append(start_state)
        path_xyz.reverse()

        # Filter out consecutive duplicates.
        unique_path = [path_xyz[0]]
        for pt in path_xyz[1:]:
            if pt != unique_path[-1]:
                unique_path.append(pt)
        path_xyz = unique_path

        # Smooth the path to reduce zigzags.
        path_xyz = self.smooth_path(path_xyz)

        # Visualize the forbidden zone and the path
        forbidden_params = self.calculate_forbidden_ellipse()
        self.visualize_path_and_forbidden_zone(path_xyz, forbidden_params)

        # Interpolate orientation linearly.
        path = []
        for i, (x, y, z) in enumerate(path_xyz):
            t = i / (len(path_xyz) - 1) if len(path_xyz) > 1 else 0
            theta = [(1 - t) * start[3 + j] + t * goal[3 + j] for j in range(3)]
            path.append([x, y, z] + theta)
        return path
