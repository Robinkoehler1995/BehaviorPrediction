#!/usr/bin/python3.6
#This script is used to create an image of the background by go through a video which should
#analyzed. It does it by going through the video frame by frane and selecting all the area
#which contains the background until all information of the background are selected

import cv2
import numpy as np
import math

path_vid = '../Ahri_V1/Data/Videos/M2U00692_Mani.avi'
cap = cv2.VideoCapture(path_vid)
ret, frame = cap.read()
rect = cv2.selectROI(frame)
out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (rect[2],rect[3]))
print(rect)
start = 100
end = 1700
for i in range(start-1):#</ skip this many frames
  ret, frame = cap.read()
for i in range(end-start):
  ret, frame = cap.read()
  if not ret:
    print('video ended')
    break
  frame = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
  cv2.imshow('frame',frame)
  out.write(frame)
  # Press Q on keyboard to  exit, this code is needed so opencv shows image correctly
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
print('done')
out.release()
cap.release()
cv2.destroyAllWindows()






