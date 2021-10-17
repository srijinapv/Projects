import numpy as np
import cv2

%matplotlib inline
from matplotlib import pyplot as plt

np.random.seed(42)

video_stream = cv2.VideoCapture('/input/video_3.mp4')

# Randomly select 30 frames
frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

# Store selected frames in an array
frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    frames.append(frame)

video_stream.release()

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
medianFrame_RGB = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2RGB)
plt.imshow(medianFrame_RGB)

# Calculate the average along the time axis
avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
avgFrame_RGB = cv2.cvtColor(avgFrame, cv2.COLOR_BGR2RGB)
plt.imshow((avgFrame_RGB))



