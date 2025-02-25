import sys
# For image processing
import numpy as np
import cv2
from utils.connect import RobotConnect
from scipy.spatial.transform import Rotation
from kortex_api.autogen.messages import VisionConfig_pb2
import time

class BasicCamera:
    """
    This class simply shows the camera feed from the robot arm
    """
    def __init__(self, robot_connection: RobotConnect):
        """
        :param robot_connection: object that has established a connection to the arm
        """
        # Save the current attached api
        self.robot_connection = robot_connection

        # You don't need a connection to access the stream itself, just the ip
        self.camera_stream = f"rtsp://{self.robot_connection.ip}/color"
        # Video capture with opencv so you can process the images
        self.video_capture = cv2.VideoCapture(self.camera_stream)

    def show_camera(self):
        """
        This function will show the camera feed from the robot arm
        """
        try:
            # Get the current frame from the video (there might be a tiny delay)
            _, image = self.video_capture.read()
            cv2.imshow("Camera feed", image)
            cv2.waitKey(1)
        except:
            print('An error has occurred, restarting camera feed')

    def end_camera(self):
        """
        Call this to end camera activities
        :return:
        """
        self.video_capture.release()
        cv2.destroyAllWindows()

    def take_picture(self):
        """
        This function will take a picture and return it
        :return: image
        """
        try:
            # Get the current frame from the video (there might be a tiny delay)
            _, image = self.video_capture.read()
            return image
        except:
            print('An error has occurred, restarting camera feed')

import threading
class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                # Display the resulting frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

from robotpy_apriltag import AprilTagDetector as apriltag
class AprilTagDetector:
    """
    This class uses the arm's camera to detect an apriltag and then calculate its center coordinates in the global frame
    """
    def __init__(self, robot_connection: RobotConnect):
        """
        :param robot_connection: object that has established a connection to the arm
        """
        # Save the current attached api
        self.robot_connection = robot_connection

        # You don't need a connection to access the stream itself, just the ip
        self.camera_stream = f"rtsp://{self.robot_connection.ip}/color"
        # Video capture with opencv so you can process the images
        self.video_capture = VideoCaptureAsync(src=self.camera_stream)
        self.video_capture.start()

        self.detector = apriltag()
        self.detector.addFamily("tag25h9")

    def detect_apriltags(self, debug: bool = False) -> list[dict[str, int|float|np.ndarray]]:
        """
        This function will look for all apriltags in the camera video stream

        :return: coordinates of all the apriltags in the global frame _relative to the center of the frame_,
                 the scale of the apriltags (for distance estimation), and the apriltag ids
        """
        attempt = 0

        while attempt < 3:
            if debug:
                print(f"Attempt {attempt}")
            
            _, image = self.video_capture.read()

            if image is None:
                raise Exception("No image found")


            # Convert the image to grayscale (required for apriltag detection)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect the apriltag
            detections = self.detector.detect(image)

            # Get the center of the apriltag in the global frame
            processed_detections = []
            for detection in detections:
                point = detection.getCenter()
                x = point.x
                y = point.y
                corner = detection.getCorner(0)
                c_x = corner.x
                c_y = corner.y
                distance = np.sqrt((c_x - x) ** 2 + (c_y - y) ** 2)
                scale = distance / 3

                image_center_x, image_center_y = image.shape[1] // 2, image.shape[0] // 2
                
                processed_detections.append({'x': x-image_center_x,
                                            'y': y-image_center_y,
                                            'z': scale,
                                            'id': detection.getId(),
                                            'image': image})
            
            if len(processed_detections) > 0:
                return processed_detections
            
            attempt += 1

        return []



    def detect_apriltag(self, id: int, debug: bool = False) -> dict[str, int|float|np.ndarray]:
        """
        This function will look for specific apriltag in the camera video stream

        :param id: id of the apriltag to be detected
        :return: coordinates of the apriltag in the global frame _relative to the center of the frame_,
                 the scale of the apriltag (for distance estimation), and the apriltag id
        """
        detected_tags = self.detect_apriltags(debug=debug)
        for tag in detected_tags:
            if debug:
                print(f"Detected tag with id {tag['id']}")
            if tag['id'] == id:
                return tag
        return None

    def detect_locate_and_display_apriltags(self, timeout: float = 30) -> None:
        """
        This function will look for all apriltags in the camera video stream and display them

        :param timeout: time to run the function for (in seconds)

        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            detections = self.detect_apriltags(debug=True)
            if len(detections) == 0:
                print('No apriltags detected')
                continue

            image = detections[0]['image']
            for detection in detections:
                x = int(detection['x']) + image.shape[1] // 2
                y = int(detection['y']) + image.shape[0] // 2
                scale = int(detection['z'])
                cv2.circle(image, (x, y), scale, (255, 0, 0), 2)
                # Also add a number of the id of the apriltag as text
                cv2.putText(image, str(detection['id']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Apriltag detection", image)
            cv2.waitKey(1)
        
        self.video_capture.stop()