import subprocess
import cv2
import numpy as np

ffmpeg_cmd = [
    "ffmpeg", "-rtsp_transport", "tcp", "-i", "rtsp://192.168.2.10/color",
    "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
]

width, height = 1280, 1530  # Set correct resolution
frame_size = width * height * 3 + 240

process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**6)



while True:
    raw_frame = process.stdout.read(frame_size)[:(width*height*3)]
    #if len(raw_frame) != frame_size:
    #    break

    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
    cv2.imshow("FFmpeg Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
process.terminate()
