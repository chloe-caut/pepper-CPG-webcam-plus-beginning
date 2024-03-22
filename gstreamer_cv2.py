import gi
import numpy as np
import cv2
from queue import Queue
from hand_detect import landmarker_and_result, draw_landmarks_on_image

"""
    COMMANDE A LANCER SUR ROBOT:
    gst-launch-0.10 -v v4l2src device=/dev/video1 ! 'video/x-raw-yuv,width=640, height=480,framerate=30/1' ! ffmpegcolorspace ! jpegenc ! rtpjpegpay ! udpsink host=192.168.1.176 port=3001

    TO INSTALL:
    pip install opencv-python mediapipe
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    sudo apt-get install libgirepository1.0-dev
    pip install PyGObject
    test
"""

import time

# GStreamer
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp

def on_new_sample(appsink):
    sample = appsink.emit('pull-sample')
    buf = sample.get_buffer()
    caps = sample.get_caps()
    data = buf.extract_dup(0, buf.get_size())
    array = np.frombuffer(data, dtype=np.uint8)
    array = array.reshape((caps.get_structure(0).get_value('height'),
                           caps.get_structure(0).get_value('width'),
                           3))
    queue.put(array)
    return Gst.FlowReturn.OK

# main
if  __name__ == "__main__":
    queue = Queue()

    Gst.init(None)

    pipeline = Gst.parse_launch(
        "udpsrc port=3001 ! application/x-rtp, encoding-name=JPEG,payload=26 ! "
        "rtpjpegdepay ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=True"
    )

    appsink = pipeline.get_by_name('sink')
    appsink.connect('new-sample', on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)

    hand_landmarker = landmarker_and_result()

    try:
        prev_time = 0

        while True:
            if not queue.empty():
                frame = queue.get()
                frame = np.copy(frame)

                hand_landmarker.detect_async(frame)
                print(hand_landmarker.result)
                frame = draw_landmarks_on_image(frame,hand_landmarker.result)

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                # Display FPS on frame
                cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('frame', frame)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pipeline.set_state(Gst.State.NULL)
        hand_landmarker.close()

        cv2.destroyAllWindows()
