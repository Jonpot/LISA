from typing import List
import cv2
import os
import numpy as np

import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    setattr(collections, "MutableSequence", collections.abc.MutableSequence)

from utils.connect import RobotConnect
from utils.vision import AprilTagDetector
from utils.arm_mover import ArmMover
from utils.verbal_interaction import VerbalInteraction
import time
import math
import json

class Position:
    def __init__(self, name: str, pose: list[float], blocked_by: list[str] = [], pos_type: str = "storage"):
        self.name = name
        self.pose = pose
        self.blocked_by = blocked_by
        self.occupied = False
        self.pos_type = pos_type

    def __str__(self):
        return f"Name: {self.name}, Pose: {self.pose}"

    def jsonify(self):
        return {"name": self.name, "pose": self.pose, "blocked_by": self.blocked_by, "occupied": self.occupied, "pos_type": self.pos_type}

class LabObject:
    def __init__(self, name: str, tag_id: int, position: str):
        self.name = name
        self.tag_id = tag_id
        self.last_storage_position = position
        self.special_handling = False
        self.current_position = position
        self.storage_position_type = "storage"

    def __str__(self):
        return f"Tag ID: {self.tag_id}, Position: {self.current_position}, Special Handling: {self.special_handling}"

    def jsonify(self):
        return {"tag_id": self.tag_id, "position": self.last_storage_position, "special_handling": self.special_handling}

class Database:
    def __init__(self):
        self.lab_objects: dict[str, LabObject] = {}
        self.positions: dict[str, Position] = {}

    def point_in_cylinder(self, point: list[float], start: list[float], end: list[float], radius: float) -> bool:
        """
        Check if a point is inside a cylinder
        :param point: the point to check
        :param start: the start of the cylinder
        :param end: the end of the cylinder
        :param radius: the radius of the cylinder
        :return: whether or not the point is in the cylinder
        """
        # Calculate the vector from the start to the end
        vec = [end[0] - start[0], end[1] - start[1], end[2] - start[2]]
        vec_mag = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        vec = [vec[0] / vec_mag, vec[1] / vec_mag, vec[2] / vec_mag]

        # Calculate the vector from the start to the point
        point_vec = [point[0] - start[0], point[1] - start[1], point[2] - start[2]]

        # Calculate the projection of the point vector onto the cylinder vector
        projection = vec[0] * point_vec[0] + vec[1] * point_vec[1] + vec[2] * point_vec[2]

        # Calculate the distance from the point to the cylinder vector
        distance = math.sqrt(point_vec[0] ** 2 + point_vec[1] ** 2 + point_vec[2] ** 2 - projection ** 2)

        return distance <= radius

    def add_position(self, name: str, pose: list[float], vi: VerbalInteraction, force: bool = False, pos_type: str = "storage"):
        # Check to see if the line drawn from this to the origin intersects with any other position
        # (this is a cylinder, not a line, because the robot has a gripper which is thick)
        blocking_positions = []
        for position in self.positions.values():
            if position.name == name:
                continue
            if self.point_in_cylinder(position.pose, pose, [0, 0, pose[2]], 0.1):
                if force or vi.ask_boolean(f"I think that this position is blocked by {position.name}. Is that correct?"):
                    blocking_positions.append(position.name)
            elif self.point_in_cylinder(pose, position.pose, [0, 0, position.pose[2]], 0.1):
                if force or vi.ask_boolean(f"I think that {position.name} is blocked by this position. Is that correct?"):
                    position.blocked_by.append(name)

        # Add the position to the database
        self.positions[name] = Position(name, pose, blocked_by=blocking_positions, pos_type=pos_type)
        
    def add_object(self, name: str, tag_id: int, position: str):
        if position not in self.positions:
            raise ValueError(f"Position {position} is not a valid position.")
        self.lab_objects[name] = LabObject(name, tag_id, position)
        self.lab_objects[name].storage_position_type = self.positions[position].pos_type
    
    def get_home_pose(self, lab_object: LabObject) -> list[float]:
        return self.positions[lab_object.last_storage_position].pose

    def jsonify(self):
        return {'lab_objects': {key: value.jsonify() for key, value in self.lab_objects.items()}, 'positions': {key: value.jsonify() for key, value in self.positions.items()}}
    
    @property
    def all_position_types(self) -> List[str]:
        """
        Returns a list of all position types in the database
        """
        return list(set([position.pos_type for position in self.positions.values()]))
    
    def open_position_of_type(self, position_type: str) -> Position:
        """
        Returns the first position of the given type that is not occupied
        """
        for position in self.positions.values():
            if position.pos_type == position_type and not position.occupied:
                return position
        return None

class Robot:
    def __init__(self, 
                 ip: str,
                 port: int,
                 credentials: tuple[str, str],
                 new_database: bool = False,
                 vi_mode: dict[str, bool] = {'speech':False, 'listening':False, 'reasoning':False, 'think_out_loud':False},
                 virtual_mode: bool = False):
        """
        :param ip: ip address of the robot
        :param port: port of the robot
        :param credentials: username and password
        :param new_database: whether or not to create a new database
        """
        # Create a connection to the robot
        if not virtual_mode:
            self.robot_connection = RobotConnect(ip, port, credentials)
            self.robot_connection.create_connection()
            self.camera = AprilTagDetector(self.robot_connection)
            self.mover = ArmMover(self.robot_connection)
        self.vi = VerbalInteraction(enable_speech=vi_mode['speech'],
                                    enable_listening=vi_mode['listening'],
                                    enable_reasoning=vi_mode['reasoning'],
                                    think_out_loud=vi_mode['think_out_loud'])

        self.holding: LabObject | None = None

        # Load/create the database
        if new_database:
            if os.path.exists("database.json"):
                os.remove("database.json")

            self.database = Database()
        else:
            if os.path.exists("database.json"):
                with open("database.json", "r") as file:
                    jsonified = json.load(file)
                    self.database = Database()
                    
                    for key, value in jsonified['positions'].items():
                        self.database.add_position(key, value['pose'], self.vi, force=True)
                        self.database.positions[key].blocked_by = value['blocked_by']
                        self.database.positions[key].occupied = value['occupied']
                        self.database.positions[key].pos_type = value['pos_type']
                    
                    for key, value in jsonified['lab_objects'].items():
                        self.database.add_object(key, value['tag_id'], value['position'])
                        self.database.lab_objects[key].special_handling = value['special_handling']
            else:
                self.database = Database()

        # Ensure object dock is in the database
        self.recent_calibration = False
        if self._get_next_empty_dock() is None and not virtual_mode:
            self.vi.speak("I don't have an object dock in my database. We must launch calibration.")
            self.calibrate_workspace()
        

        # Move to home position
        if not virtual_mode:
            home_status = self.mover.move_to_pose(self.mover.home, blocking=True)
            if not home_status:
                self.vi.speak("Failed to move to home position, something is wrong. Aborting.")
                self.robot_connection.close_connection()
                exit()

            self.mover.open_gripper()

    def _get_next_empty_swap_position(self) -> Position:
        """
        Get the next empty swap position in the database
        """
        for position in self.database.positions.values():
            if position.pos_type == "swap" and not position.occupied:
                return position
        return None

    def _get_next_empty_dock(self) -> Position:
        """
        Get the next empty object dock in the database
        """
        for position in self.database.positions.values():
            if position.pos_type == "object_dock" and not position.occupied:
                return position
        return None

    def exit(self):
        # Save the database
        with open("database.json", "w") as file:
            json.dump(self.database.jsonify(), file)

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
        object_name = self.vi.ask("What is this object?")
        special_handling = self.vi.ask("Does this object require special handling? (y/n)")
        special_handling = True if 'y' in special_handling.lower() else False

        # Add the object to the database
        self.database.add_position(f'{object_name}_pos', pose, self.vi)
        self.database.add_object(object_name, tag_id, f'{object_name}_pos')
        self.database.lab_objects[object_name].special_handling = special_handling

    def scan_and_populate_database_naive(self):
        """
        This function will scan the environment for apriltags and populate the database
        """
        
        # Aggregate seen tags from database
        seen_tags = set()
        for lab_object in self.database.lab_objects.values():
            seen_tags.add(lab_object.tag_id)

        # Scan the environment to add new tags
        unseen_tag = self.mover.scan_for_unseen_apriltag(self.camera, seen_tags)
        while unseen_tag is not None:
            # Approach the tag, but not too close
            success = self.mover.approach_apriltag_detection(self.camera, unseen_tag, threshold=425)

            # Add the object to the database
            if success:
                self._add_object_to_database(unseen_tag['id'], self.mover.robot_position)
                seen_tags.add(unseen_tag['id'])

            unseen_tag = self.mover.scan_for_unseen_apriltag(self.camera, seen_tags)

        self.camera.video_capture.stop()

    def identify_calibration_tag(self) -> int:
        """
        This function will identify the calibration tag
        """
        self.vi.speak("Please show me the tag of the calibration object.")
        while True:
            detections = []
            while len(detections) == 0:
                detections = self.camera.detect_apriltags()
                # Remove tags already in database
                detections = [detection for detection in detections if detection['id'] not in [tag.tag_id for tag in self.database.lab_objects.values()]]
                if len(detections) == 0:
                    continue
                else:
                    print(detections)

            # Approach the tag, but not too close
            success = self.mover.approach_apriltag_detection(self.camera, detections[0], threshold=425)
            if not success:
                self.vi.think("Failed to approach the tag, trying again.")
            else:
                self.mover.waggle_gripper()
                if self.vi.ask_boolean("Is this the calibration object?"):
                    return detections[0]['id']
                else:
                    continue

    def calibrate_workspace(self):
        """
        This function will have the robot collaborate with a lab assistant to calibrate the workspace
        """
        # Move home
        self.mover.move_to_pose(self.mover.home)

        calibration_tag = self.identify_calibration_tag()

        while self.vi.ask_boolean("Would you like to add a new position to the database?"):
            if not self.vi.ask_boolean("Great! Go ahead and place the calibration object on the position you want to save, then let me know when you're ready to continue."):
                self.vi.speak("Hm, sounds like you don't want to continue. I'm aborting the process.")
                break
            
            self.vi.speak("Alright, I'm going to look for it now.")
            detection = self.mover.scan_for_apriltag(self.camera, calibration_tag)
            while detection is None:
                self.vi.think("I couldn't find the calibration object. I'm going to try again.")
                detection = self.mover.scan_for_apriltag(self.camera, calibration_tag)

            self.vi.think("I found the calibration object. I'm going to approach it now.")
            self.mover.approach_apriltag_detection(self.camera, detection, threshold=425)
            self.mover.waggle_gripper()
            position_name = self.vi.ask("What is the name of this position?")
            while position_name in self.database.positions:
                position_name = self.vi.ask("That position is already in the database. Please provide a different name.")
            position_type = self.vi.ask_position_type("What type of position is this? (object_dock, swap, storage, etc.)")
            self.vi.think(f"This is the {position_name}, I'm updating my internal memory of this position.")
            self.database.add_position(position_name, self.mover.robot_position, self.vi, pos_type = position_type)
        
        self.recent_calibration = True

    def scan_and_populate_database(self):
        """
        Works alongside a lab assistant to populate the database
        """
        # Move home
        self.mover.move_to_pose(self.mover.home)

        if self.vi.ask_boolean("Would you like to create a new database?"):
            self.database = Database()

        if not self.recent_calibration and self.vi.ask_boolean("Would you like to calibrate the workspace?"):
            self.calibrate_workspace()
        
        while not self.vi.ask_boolean("Great! I've added all the positions to my database. Go ahead and populate the lab with objects and let me know when you're ready to continue."):
            time.sleep(30)

        self.vi.speak("Alright, I'm going to look for objects now.")

        # Go to every position and scan for objects
        for position in self.database.positions:
            self.mover.move_to_pose(self.mover.home)
            self.vi.think(f"Moving to {position}.")
            self.mover.move_to_pose(self.database.positions[position].pose)
            self.vi.think(f"Scanning {position} for objects.")
            detections = self.camera.detect_apriltags()
            for detection in detections:
                self.vi.think(f"Detected object with tag id {detection['id']}.")
                if self.vi.ask_boolean("Would you like to add this object to the database?"):
                    name = self.vi.ask("What is the name of this object?")
                    while name in self.database.lab_objects:
                        name = self.vi.ask("That object is already in the database. Please provide a different name.")
                    special_handling = self.vi.ask_boolean("Does this object require special handling?")
                    self.database.add_object(name, detection['id'], position)
                    self.database.lab_objects[name].special_handling = special_handling
                    self.database.positions[position].occupied = True
                else:
                    self.vi.think("Okay, I won't add this object to the database.")

        self.vi.speak("Great! I've added all the objects to my database.")

        self.mover.move_to_pose(self.mover.home)

        # Save the database
        with open("database.json", "w") as file:
            json.dump(self.database.jsonify(), file)

    def _get_object_from_db(self,
                            name: str|None = None,
                            position: str|None = None,
                            pose: list[float]|None = None,
                            id: int|None = None) -> LabObject:
        """
        Helper function to get an object from the database
        """
        if name is not None:
            # Naive search requiring perfect match
            if name in self.database.lab_objects:
                self.vi.speak(f"Certainly, I have an object named {name} in my database. I'll get it for you.")
                return self.database.lab_objects[name]
            elif self.vi.reasoning:
                self.vi.think(f"I don't have an object the exact name '{name}' in my database. Let me reason about what they might mean.")
                name = self.vi.reason(f"A user is asking for an object named {name}. My database contains the following objects: {', '.join(self.database.lab_objects.keys())}. Which object should I retrieve? Return only the exact name of the object and no other text.")
                if name in self.database.lab_objects:
                    self.vi.think(f"Based on reasoning, I believe the user is asking for the object '{name}'.")
                    self.vi.speak(f"Sounds like you want {name}. I'll get it for you.")
                    return self.database.lab_objects[name]
        elif position is not None:
            for lab_object_name, lab_object in self.database.lab_objects.items():
                if self.database.lab_objects[lab_object_name].last_storage_position == position:
                    return lab_object
        elif pose is not None:
            for lab_object_name, lab_object in self.database.lab_objects.items():
                if self.database.positions[lab_object_name].pose == pose:
                    return lab_object
        elif id is not None:
            for _, lab_object in self.database.lab_objects.items():
                if lab_object.tag_id == id:
                    return lab_object

        return None

    def prepare_to_pickup(self, lab_object: LabObject) -> None:
        """
        Sets speeds for the robot to pick up an object
        """
        if lab_object.special_handling:
            self.mover.set_default_cartesian_constraints(2)
        else:
            self.mover.set_default_cartesian_constraints(13)

        self.holding = lab_object


    def set_gripper_empty(self) -> None:
        """
        Sets the gripper to empty
        """
        self.mover.set_default_cartesian_constraints(13)
        self.holding = None


    def remove_blocking_object(self, blocking_object: LabObject, objective_position: Position) -> None:
        """
        Similar to retrieve_from_shelf, but doesn't perform a general search
        if the object isn't found
        """
        # Move to the object
        self.mover._move_to_current_position()
        objective_position = self.database.get_home_pose(blocking_object)
        self.mover.move_to_pose(objective_position)

        # Verify the object is there
        time.sleep(1)
        self.vi.think(f"Checking for object with tag id {blocking_object.tag_id}")
        detection = self.camera.detect_apriltag(blocking_object.tag_id, debug= True)
        if detection is None:
            self.vi.think("Object not found at the expected location.")
            return

        self.prepare_to_pickup(blocking_object)
        self.mover.move_apriltag_detection(self.camera, detection, objective_pose=objective_position.pose)
        self.set_gripper_empty()

        # Mark this position as unoccupied
        self.database.positions[blocking_object.current_position].occupied = False

        # Mark the objective position as occupied
        self.database.positions[objective_position.name].occupied = True

        # And the object's current position to
        blocking_object.current_position = objective_position.name


    def smart_move_to_position(self, objective_position: Position) -> tuple[dict[LabObject, Position], dict[LabObject, Position]]:
        self.mover._move_to_current_position()
        held_blocking_positions: list[Position] = []
        held_blocking_objects: dict[LabObject, Position] = {}

        swapped_blocking_positions: list[Position] = []
        swapped_blocking_objects: dict[LabObject, Position] = {}

        blocking_positions: list[Position] = []
        if len(objective_position.blocked_by) > 0:
            self.vi.think("This position can be obscured by other objects. I'll check to see if it's clear.")
            for potential_blocker in objective_position.blocked_by:
                if self.database.positions[potential_blocker].occupied:
                    blocking_positions.append(self.database.positions[potential_blocker])

        if len(blocking_positions) > 0:
            # sort them by the length of their blocked_by list, so we remove
            # the front-most blocker first
            blocking_positions.sort(key=lambda x: len(x.blocked_by))
            # We have blocking positions, but we might be able to temporarily move them to a swap position
            while self._get_next_empty_swap_position() is not None and len(blocking_positions) > 0:
                blocking_position = blocking_positions.pop(0)
                swap_position = self._get_next_empty_swap_position()
                self.vi.think(f"Object {blocking_position.name} is blocking the way. I'll move it to swap position {swap_position.name}.")
                # Get the object in the blocking position
                blocking_object: LabObject = self._get_object_from_db(position=blocking_position.name)
                if blocking_object is None:
                    self.vi.think("Something has gone wrong and this position shouldn't have been marked as occupied")
                    self.database.positions[blocking_position.name].occupied = False
                    continue

                #  Move the object to the swap position
                self.move_object(blocking_position, blocking_object, swap_position)

                # Mark this position as unoccupied
                self.database.positions[blocking_position.name].occupied = False

                # Mark the swap position as occupied
                self.database.positions[swap_position.name].occupied = True

                # And the object's current position to swap_position
                blocking_object.current_position = swap_position.name

                # Add it to the list of swapped objects
                swapped_blocking_objects[blocking_object] = blocking_position
                swapped_blocking_positions.append(blocking_position)
                
            swapped_blocking_objects = {key: value for key, value in reversed(swapped_blocking_objects.items())}

            if len(blocking_positions) > 0:
                # We have blocking positions, but we can't move them to a swap position
                # We will remove them one by one by bringing them to the object dock at the next available object dock
                object_dock = self._get_next_empty_dock()
                if object_dock is None:
                    self.vi.think("I can't find an empty object dock to put this object. I can't proceed.")
                    return held_blocking_objects, swapped_blocking_objects
                
                # Now convert the positions into the object names at those positions
                for position in held_blocking_positions:
                    held_blocking_objects[self._get_object_from_db(position=position)] = position

                # Now retrieve the objects blocking the way, one by one
                for blocking_object, position in held_blocking_objects.items():
                    self.vi.think(f"Object {blocking_object.name} is blocking the way. I'll remove it.")
                    self.vi.ask_boolean(f"Please remove the object {blocking_object.name} from the path, and let me know when you're ready.")
                    self.remove_blocking_object(blocking_object)
                
                self.vi.think("The way is clear now. I'll proceed to the object's position.")

        #print(f"moving to pose! {objective_position.pose}")
        self.mover.move_to_pose(objective_position.pose)

        # Reverse the blocking objects so we put them back in the right order (back to front)
        held_blocking_objects = {key: value for key, value in reversed(held_blocking_objects.items())}
        return held_blocking_objects, swapped_blocking_objects

    def smart_move_to_object(self, lab_object: LabObject) -> tuple[dict[LabObject, Position], dict[LabObject, Position]]:
        return self.smart_move_to_position(self.database.positions[lab_object.current_position])
        

    def retrieve_from_shelf(self, object_name: str, objective_position: Position):
        """
        This function will retrieve an object from a shelf according to the database
        """
         # Get the object from the database
        lab_object: LabObject = self._get_object_from_db(name=object_name)
        if lab_object is None:
            self.vi.speak(f"Hmm, {object_name} isn't in the database. I can't retrieve it.")
            return

        # If the object isn't currently at it's home position, fail (not implemented behavior)
        if lab_object.current_position != lab_object.last_storage_position:
            self.vi.speak(f"Sorry, I can't retrieve {object_name} right now. It's in use by another scientist.")

        self.move_object(lab_object.current_position, lab_object, objective_position)

    def return_to_shelf(self, object_name: str | None):
        """
        This function will return an object to the shelf according to the database
        """
        # Get the object from the database
        lab_object: LabObject = self._get_object_from_db(name=object_name)
        if lab_object is None:
            self.vi.speak(f"Hmm, {object_name} isn't in the database. I can't return it.")
            return

        self.move_object(lab_object.current_position, lab_object, lab_object.last_storage_position)


    def move_object(self, start_position: Position | None = None, object_to_move: LabObject | None = None, end_position: Position | None = None):
        """
        This function will return an object to the shelf
        """
        held_blocking_objects: dict[LabObject, Position] = {}
        swapped_blocking_objects: dict[LabObject, Position] = {}

        if start_position is None and object_to_move is None:
            self.vi.speak("I can't move an unspecified object if I don't know where to look.")
            return
        
        if start_position is None:
            # Get the start position from the database
            start_position = self.database.positions[object_to_move.current_position]

        if object_to_move is None:
            # Go to start position, detect apriltags to figure out what is being returned
            new_held_blocking_objects, new_swapped_blocking_objects = self.smart_move_to_position(start_position)
            held_blocking_objects |= new_held_blocking_objects
            swapped_blocking_objects |= new_swapped_blocking_objects

            # Detect apriltags
            time.sleep(0.5)
            detections = self.camera.detect_apriltags()
            while len(detections) == 0:
                self.vi.think("No objects detected at the object dock, returning to home")
                detections = self.camera.detect_apriltags()

            self.mover.move_home_from_dock()
            # Get the object that is being returned
            detection_id = detections[0]['id']

            # Get the object from the database
            object_to_move: LabObject = self._get_object_from_db(id=detection_id)
            if object_to_move is None:
                self.vi.speak("Object not found in the database")
                return

            self.vi.think(f"Detected object with tag id {detection_id}, which is {object_to_move.name}.")

        if end_position is None:
            # Assume we're returning the object to a position consistent with it's storage type
            end_position = self.database.open_position_of_type(object_to_move.storage_position_type)
            if end_position is None:
                # then just return it to where we got it from
                end_position = self.database.positions[object_to_move.last_storage_position]

        # Verify the return position is not occupied if its not a dock
        if end_position.pos_type != "object_dock":
            new_held_blocking_objects, new_swapped_blocking_objects = self.smart_move_to_position(end_position)
            held_blocking_objects |= new_held_blocking_objects
            swapped_blocking_objects |= new_swapped_blocking_objects

            detections = self.camera.detect_immediate_apriltags()
            occupied = False
            if len(detections) > 0:
                self.vi.think("Return position might be occupied by another object, updating the database and backtracking")
                occupied = True
                detection = detections[0]
                self.vi.think(f"Detected occupying tag with id {detection['id']}")
            
            while occupied:
                # Get the pose of the object that was actually there
                occupying_object: LabObject = self._get_object_from_db(id=detection['id'])
                if occupying_object is None:
                    self.vi.think("New object detected, updating the database")
                    self._add_object_to_database(detection['id'], end_position.pose)
                    self.vi.think("Database updated, but I don't know where to put this object. Returning to dock.")
                    self.mover._move_to_current_position()
                    self.mover.move_to_pose(self.mover.home)
                    self.vi.speak("While I was returning an object, I found a new object where this one should go. I added that one to my databse, but I don't know where to put this one. Please help me.")
                    return

                # Update the database
                prior_pos = occupying_object.current_position
                occupying_object.current_position = end_position.name
                self.database.positions[end_position.name].occupied = True
                self.database.positions[prior_pos].occupied = False
                end_position = self.database.positions[prior_pos]

                # Visit the prior pos to see if it's still occupied
                self.mover._move_to_current_position()
                new_held_blocking_objects, new_swapped_blocking_objects = self.smart_move_to_position(end_position)
                held_blocking_objects |= new_held_blocking_objects
                swapped_blocking_objects |= new_swapped_blocking_objects

                detections = self.camera.detect_apriltags()
                if len(detections) == 0:
                    occupied = False
                else:
                    detection = detections[0]
                    self.vi.think(f"Detected tag with id {detection['id']}")
                    continue

        # At this point, we know the object's target position (even if that was updated) is not occupied
        # Move to the object from the start position to the target position
        self.vi.think(f"Moving to the start position ({start_position.name}) to retrieve the object.")
        if len(held_blocking_objects) > 0 and start_position.pos_type == "object_dock":
            self.vi.ask_boolean(f"Please place the object {object_to_move.name} back at {start_position.name} and let me know when you're ready.")
        new_held_blocking_objects, new_swapped_blocking_objects = self.smart_move_to_position(start_position)
        held_blocking_objects |= new_held_blocking_objects
        swapped_blocking_objects |= new_swapped_blocking_objects
        
        detection = self.camera.detect_apriltag(object_to_move.tag_id, debug=True)
        while detection is None:
            detection = self.camera.detect_apriltag(object_to_move.tag_id, debug=True)
        
        self.prepare_to_pickup(object_to_move)
        self.mover.move_apriltag_detection(self.camera, detection, objective_pose=end_position.pose)
        self.set_gripper_empty()

        # Mark this position as occupied
        if end_position.pos_type != "object_dock": # object docks can't be occupied, they are permanent sinks
            self.database.positions[end_position.name].occupied = True

        # And the object's current position to the position it was returned to
        object_to_move.current_position = end_position.name

        # If we removed a blocking object and gave it to the lab tech, we need to put it back
        if len(held_blocking_objects) > 0:
            self.vi.speak(f"I've retrieved the object {object_to_move.name}, but I need to put the blocking objects back.")
        while len(held_blocking_objects) > 0:
            blocking_object, position = held_blocking_objects.popitem()
            self.vi.speak(f"Please place the object {blocking_object.name} back at the object dock.")
            self.vi.ask_boolean("Let me know when you're ready.")
            self.move_object(start_position, blocking_object, position)

        # Once we've put the blocking objects back, we need to put the swapped blocking objects back
        if len(swapped_blocking_objects) > 0:
            self.vi.speak(f"I've retrieved the object {object_to_move.name}, but I need to put the swapped blocking objects back.")
        while len(swapped_blocking_objects) > 0:
            blocking_object, position = swapped_blocking_objects.popitem()
            self.vi.think(f"Returning object {blocking_object.name} to position {position.name}.")
            self.move_object(blocking_object.current_position, blocking_object, position)