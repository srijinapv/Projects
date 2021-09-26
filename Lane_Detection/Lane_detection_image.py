import matplotlib.pylab as plt
import cv2
import numpy as np

#Function to mask region other than region of interest
def region_interest(pic, sides):
    mask = np.zeros_like(pic)
    #number_of_channels = pic.shape[2]
    matching_mask_number_channel = 255
    cv2.fillPoly(mask, sides, matching_mask_number_channel)
    masked_image = cv2.bitwise_and(pic, mask)
    return masked_image

# Function to draw the line
def draw_line(pic,lines):
    pic = np.copy(pic)
    blank_image = np.zeros((pic.shape[0], pic.shape[1], 3) , dtype = np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0), thickness=3)
    pic = cv2.addWeighted(pic,0.8,blank_image,1,0.0)
    return pic


image = cv2.imread("/Users/srijina/360DIGITMG_MY_LEARNING/Project_Datascience_online/Lane_detection/road.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print(image.shape)
image_height = image.shape[0]
image_width = image.shape[1]

plt.imshow(image)
plt.show()

region_of_interest = [(0,image_height),(image_width/2 , image_height/1.68),(image_width,image_height)]

#converting the resulting image to gray scale to find the edges
gray_result_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#Edges are detected using canny edge detector with lowe threshold of 100 and upper thershold of 200
canny_image = cv2.Canny(gray_result_image, 100, 200)

plt.imshow(canny_image)
plt.show()

resulting_image = region_interest(canny_image, np.array([region_of_interest], np.int32))
plt.imshow(resulting_image)
plt.show()

#Draw the line through the edges to highlight the lanes using probabilistic hough line transform
lines = cv2.HoughLinesP(resulting_image , rho=6 , theta= np.pi/60 , threshold=160,
                       lines = np.array([]) , minLineLength=40 , maxLineGap=25 )

highlighting_lanes = draw_line(image,lines)
plt.imshow(highlighting_lanes)
plt.show()
