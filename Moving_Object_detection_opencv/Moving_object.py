import numpy as np
import cv2
import argparse

def background(path):
    video_input =cv2.VideoCapture(path)
    frame =video_input.get(cv2.CAP_PROP_FRAME_COUNT)*np.random.uniform(size = 50)
    number_frame = []
    for i in frame:
        video_input.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, captured_frame = video_input.read()
        number_frame.append(captured_frame)
    # calculate the median
    median_frame = np.median(number_frame, axis=0).astype(np.uint8)

    return median_frame

