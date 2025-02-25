import cv2
import os
import numpy as np
from utils.connect import RobotConnect
from utils.vision import AprilTagDetector
from utils.arm_mover import ArmMover
import time
import math
import json

class LabObject:
    def __init__(self, tag_id: int, pose: list[float]):
        self.tag_id = tag_id
        self.pose = pose
        self.special_handling = False

    def __str__(self):
        return f"Tag ID: {self.tag_id}, Pose: {self.pose}, Special Handling: {self.special_handling}"

    def jsonify(self):
        return {"tag_id": self.tag_id, "pose": self.pose, "special_handling": self.special_handling}

class Robot:
    def __init__(self, 
                 ip: str,
                 port: int,
                 credentials: tuple[str, str],
                 new_database: bool = False):
        """
        :param ip: ip address of the robot
        :param port: port of the robot
        :param credentials: username and password
        :param new_database: whether or not to create a new database
        """
        # Create a connection to the robot
        self.robot_connection = RobotConnect(ip, port, credentials)
        self.robot_connection.create_connection()
        self.camera = AprilTagDetector(self.robot_connection)
        self.mover = ArmMover(self.robot_connection)

        # Load/create the database
        if new_database:
            if os.path.exists("database.json"):
                os.remove("database.json")

            self.database = {}
        else:
            if os.path.exists("database.json"):
                with open("database.json", "r") as file:
                    jsonified = json.load(file)
                    self.database = {key: LabObject(value["tag_id"], value["pose"]) for key, value in jsonified.items()}
            else:
                self.database: dict[str, LabObject] = {}

        # Move to home position
        home_status = self.mover.move_to_pose(self.mover.home, blocking=True)
        if not home_status:
            print("Failed to move to home position, something is wrong. Aborting.")
            self.robot_connection.close_connection()
            exit()

        self.mover.open_gripper()

    def exit(self):
        # Save the database
        with open("database.json", "w") as file:
            jsonified = {key: value.jsonify() for key, value in self.database.items()}
            json.dump(jsonified, file)

        # Close the connection
        self.camera.video_capture.stop()
        self.robot_connection.close_connection()

    def _add_object_to_database(self,
                                tag_id: int,
                                pose: list[float]
                                ):
        """
        Helper function to add an object to the database
        """
        # Inquire about this object
        self.mover.waggle_gripper()
        object_name = input("What is this object? ")
        special_handling = input("Does this object require special handling? (y/n) ")
        special_handling = True if special_handling.lower() == "y" else False

        # Add the object to the database
        self.database[object_name] = LabObject(tag_id, pose)
        self.database[object_name].special_handling = special_handling

    def scan_and_populate_database(self):
        """
        This function will scan the environment for apriltags and populate the database
        """
        
        # Aggregate seen tags from database
        seen_tags = set()
        for tag in self.database.values():
            seen_tags.add(tag.tag_id)

        # Scan the environment to add new tags
        unseen_tag = self.mover.scan_for_unseen_apriltag(self.camera, seen_tags)
        while unseen_tag is not None:
            # Approach the tag, but not too close
            success = self.mover.approach_apriltag_detection(self.camera, unseen_tag, threshold=425)

            # Add the object to the database
            if success:
                self._add_object_to_database(unseen_tag['id'], self.mover.robot_position())
                seen_tags.add(unseen_tag['id'])

            unseen_tag = self.mover.scan_for_unseen_apriltag(self.camera, seen_tags)

        self.camera.video_capture.stop()

    def _get_object_from_db(self,
                            name: str|None = None,
                            pose: list[float]|None = None,
                            id: int|None = None) -> LabObject:
        """
        Helper function to get an object from the database
        """
        if name is not None:
            return self.database[name]
        elif pose is not None:
            for object in self.database.values():
                if object.pose == pose:
                    return object
        elif id is not None:
            for object in self.database.values():
                if object.tag_id == id:
                    return object

        return None

    def retrieve_from_shelf(self, object_name: str):
        """
        This function will retrieve an object from a shelf according to the database
        """
         # Get the object from the database
        object: LabObject = self._get_object_from_db(name=object_name)
        if object is None:
            print("Object not found in the database")
            return

        # Move to the object
        self.mover._move_to_current_position()
        self.mover.move_to_pose(object.pose)

        # Verify the object is there
        time.sleep(1)
        print(f"Checking for object with tag id {object.tag_id}")
        detection = self.camera.detect_apriltag(object.tag_id, debug= True)
        if detection is None:
            print("Object not found at the expected location, performing a more general search")
            self.mover.find_and_retrieve_apriltag(self.camera, object.tag_id)
        else:
            print("Object found at the expected location")
            self.mover.retrieve_apriltag_detection(self.camera, detection)

        self.camera.video_capture.stop()

    def return_to_shelf(self):
        """
        This function will return an object to the shelf
        """

        # Go to object dock, detect apriltags to figure out what is being returned
        self.mover._move_to_current_position()
        self.mover.move_to_object_dock()

        # Detect apriltags
        detections = self.camera.detect_apriltags()
        self.mover.move_home_from_dock()
        if len(detections) == 0:
            print("No objects detected at the object dock, returning to home")
            return
        
        # Get the object that is being returned
        detection_id = detections[0]['id']

        # Get the object from the database
        object: LabObject = self._get_object_from_db(id=detection_id)
        if object is None:
            print("Object not found in the database")
            return

        # Verify the object's home is not occupied
        self.mover._move_to_current_position()
        self.mover.move_to_pose(object.pose)
        detections = self.camera.detect_apriltags()
        occupied = False
        if len(detections) > 0:
            print("Object's home might be occupied by another object, updating the database and backtracking")
            for detection in detections:
                # Check if the detection is in the center of the image and close (otherwise might just be in background)
                print(f"Detected tag with id {detection['id']}, detection['x'] {detection['x']} detection['y'] {detection['y']} detection['z'] {detection['z']}")
                if abs(detection['x']) < 50 and abs(detection['y']) < 50 and detection['z'] > 10:
                    occupied = True
                    detection = detections[0]
                    print(f"Detected tag with id {detection['id']}")
                    break
            if not occupied:
                print("Found detections, but none were in the center of the image and close. Continuing to return.")
        
        while occupied:
            # Get the pose of the object that was actually there
            occupying_object: LabObject = self._get_object_from_db(id=detection['id'])
            if occupying_object is None:
                print("New object detected, updating the database")
                self._add_object_to_database(detection['id'], object.pose)
                print("Database updated, but I don't know where to put this object. Returning to dock.")
                self.mover._move_to_current_position()
                self.mover.move_to_pose(self.mover.home)
                return

            # Update the database
            prior_pos = occupying_object.pose
            occupying_object.pose = object.pose
            object.pose = prior_pos

            # Visit the prior pos to see if it's still occupied
            self.mover._move_to_current_position()
            self.mover.move_to_pose(object.pose)
            detections = self.camera.detect_apriltags()
            if len(detections) == 0:
                occupied = False
            else:
                detection = detections[0]
                print(f"Detected tag with id {detection['id']}")
                continue

        # At this point, we know the object's home (even if that was updated) is not occupied
        # Move to the object from the object dock to this location
        self.mover._move_to_current_position()
        self.mover.move_to_pose(self.mover.home)
        self.mover.open_gripper()
        self.mover.move_to_object_dock()

        detection = self.camera.detect_apriltag(detection_id, debug=True)
        self.mover.approach_apriltag_detection(self.camera, detection)
        self.mover.close_gripper()
        self.mover.move_relative_to_tcp([0, 0, self.mover.correction_up_amount], blocking=True)
        
        self.mover.move_home_from_dock()
        self.mover.move_to_pose(object.pose)

        # Move forward ~10cm to ensure the object is placed on the shelf
        self.mover.move_relative_to_tcp([0, 0, self.mover.correction_up_amount], blocking=True)
        self.mover.move_relative_to_tcp([self.mover.correction_forward_amount, 0, 0], blocking=True)

        self.mover.open_gripper()
        self.mover.move_relative_to_tcp([-self.mover.correction_forward_amount, 0, 0], blocking=True)
        self.mover.move_to_pose(self.mover.home)

        self.camera.video_capture.stop()