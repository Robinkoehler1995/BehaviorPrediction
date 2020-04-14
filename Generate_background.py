#!/usr/bin/python3.6
#This script is used to create an image of the background by go through a video which should
#analyzed. It does it by going through the video frame by frane and selecting all the area
#which contains the background until all information of the background are selected

import cv2
import numpy as np
import math

#aplies operation "op" to an image "img" with a kernel size of "size"
def morph(img,op=cv2.MORPH_CLOSE,size=5):
  kernel = np.ones((size,size),np.uint8)
  img = cv2.morphologyEx(img, op, kernel)
  return img

#draws a rectange of the shape "r" onto te image "img"
def drawRect(img,r):
  r1 = (r[0],r[1])
  r2 = (r[0]+r[2],r[1]+r[3])
  cv2.rectangle(img,r1,r2,255,-1)

#setting up video path
path_vid = 'test.avi'
#path_vid = '../Data/Videos/M2U00441.MPG'
#path_vid = '../Data/Videos/M2U00442.MPG'
#path_vid = '../Data/Videos/AgonisticGal3.MPG'
#path_vid = '../Data/Videos/wt.mp4'
#path_vid = '../Data/Videos/KO.mp4'
 
#init video reader
cap = cv2.VideoCapture(path_vid)

#init background model "backgorund"
ret, frame = cap.read()
background = np.zeros(frame.shape[:2], np.uint8)
background_color = np.zeros(frame.shape, np.uint8)
#selects how ofter an area of the frame is removed from the frame to build the background 
animals = 4
#starting frame
frameNumber = 1
for i in range(1):#</ skip this many frames
  ret, frame = cap.read()

#if valid is all white (all elements are 255) the background model is finished
valid = np.zeros(frame.shape[:2], np.uint8)

while(cap.isOpened()):
  #print frame nummber everz 100 iterations
  if frameNumber%100==0:
    print(frameNumber)

  #read current frame
  ret, frame = cap.read()
  #if there is a current frame contiune
  if ret:

    #preprocessing the current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #every 200th frame is chosen to add information to the background model
    if frameNumber%20 == 0:
      #contains the area of the infromation which are added to the background model
      valid_current = np.zeros(gray.shape[:2], np.uint8)
      #"roi" contains the information which are not yet part of the background
      roi = cv2.bitwise_and(gray,cv2.bitwise_not(valid))
      roi = cv2.bitwise_or(roi,background)

      for i in range(animals):
        #select area which should not be added to the background model. THis is done by the user
        rect = cv2.selectROI(roi)
        if rect != (0,0,0,0):
          drawRect(valid_current,rect)
      
      #determine the infromation which are added to the background which were selected by the user
      valid_current = cv2.bitwise_not(valid_current)
      valid_add = cv2.bitwise_and(valid_current,cv2.bitwise_not(valid))
      valid = cv2.bitwise_or(valid,valid_add)

      #add the information to the background model
      background_add = cv2.bitwise_and(valid_add,gray)
      background = cv2.bitwise_or(background,background_add)

      #add the information to the background model in color
      background_color_add = cv2.bitwise_and(frame,frame,mask=valid_add)
      background_color = cv2.bitwise_or(background_color,background_color_add)

      #print progress
      print(np.count_nonzero(valid),valid.shape[0]*valid.shape[1])
      #if vaild does not contain any elements with 0 the background model is finisehd and it is saved
      if np.count_nonzero(valid) == valid.shape[0]*valid.shape[1]:
        output_base = path_vid.split('.')[0]
        cv2.imwrite(output_base+'_background.png',background)
        cv2.imwrite(output_base+'_background_color.png',background_color)
        print('succ writing: '+output_base)
        break

      #show progress
      cv2.imshow('v',valid)
      #cv2.imshow('c',valid_current)
      #cv2.imshow('a',valid_add)


      # Press Q on keyboard to  exit, this code is needed so opencv shows image correctly
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frameNumber += 1
  # if the video reader has no current frame anymore the video is over and the loop is escaped
  else: 
    print('video ended before background was produced')
    break
#the video reader is released and all open videos of opencv are closed
cap.release()
cv2.destroyAllWindows()






