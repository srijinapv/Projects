import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = '/images'
images = []
p_name = []
mylist = os.listdir(path)
print(mylist)

for img in mylist :
    current_img = cv2.imread(f'{path}/{img}')
    images.append(current_img)
    p_name.append(os.path.splitext(img)[0])
print(p_name)

def faceEncoder(images):
    encode_list = []
    for picture in images:
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        encoder = face_recognition.face_encodings(picture)[0]
        encode_list.append(encoder)
    return encode_list
print(faceEncoder(images))