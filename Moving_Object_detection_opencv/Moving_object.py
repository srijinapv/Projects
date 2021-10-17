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

sample_frame=frames[0]
sample_frame_RGB = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
plt.imshow((sample_frame_RGB))

grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
grayMedianFrame_RGB =cv2.cvtColor(grayMedianFrame, cv2.COLOR_GRAY2RGB)
plt.imshow((grayMedianFrame))

graySample=cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
graySample_RGB = cv2.cvtColor(graySample, cv2.COLOR_GRAY2RGB)
plt.imshow((graySample_RGB))

dframe = cv2.absdiff(graySample, grayMedianFrame)
dframe_RGB = cv2.cvtColor(dframe, cv2.COLOR_GRAY2RGB)
plt.imshow((dframe))

blurred = cv2.GaussianBlur(dframe, (11,11), 0)
blurred_RGB = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
plt.imshow((blurred_RGB))

ret, tframe= cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
tframe_RGB = cv2.cvtColor(tframe, cv2.COLOR_GRAY2RGB)
plt.imshow((tframe_RGB))

writer = cv2.VideoWriter("output.mp4",
                         cv2.VideoWriter_fourcc(*"MP4V"), 30,(640,480))

#Create a new video stream and get total frame count

video_stream = cv2.VideoCapture('/video_3.mp4')


total_frames=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
total_frames

frame_count=0
while(frame_count < total_frames-1):

    frame_count+=1
    ret, frame = video_stream.read()

    # Convert current frame to grayscale
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and
    # the median frame
    dframe = cv2.absdiff(gframe, grayMedianFrame)
    # Gaussian
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)


    if frame_count % total_frames == 0 or frame_count == 1:
        frame_diff_list = []

    #Thresholding to binarise
    ret, thres = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
    dilate_frame = cv2.dilate(thres, None, iterations=2)
    frame_diff_list.append(dilate_frame)
    if len(frame_diff_list) == total_frames:
        sum_frames = sum(frame_diff_list)
        contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Detected Objects', frame)
        out.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

#Release video object
video_stream.release()
writer.release()




