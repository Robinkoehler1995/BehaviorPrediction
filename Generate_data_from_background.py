#!/usr/bin/python3.6
#this script uses the background model created by "generate_background.py" to create training data which is used 
#to train the first pixel wise prediction of the tool descripted in the thesis "Optimization in the analysis of social interaction test in rodents"
#It uses the background model and subtracts it from the current frame to get all the information which are in the foreground.
#This script also updates the background model accordingly to ensure high quality output
import cv2
import numpy as np
import math, sys
from tqdm import tqdm
import net_handler as nh

#this method stops the script until to look at an image
def lucidDreaming():
    while(True):
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print('its getting dark')
            return True
        if k == 32:
            return False

#this method close all open windows and quits the script
def onExit():
    print('its getting dark')
    cv2.destroyAllWindows()
    sys.exit()

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

#defines the shape of the output of the whole frame
whole_shape = (128,128)
#defines the storage valiable all images are saveind in
whole_X = list()
whole_y = list()

#defines the shape of the output of the seperate contour found in the foreground
fragment_shape = (128,128)
#defines the storage valiable all images are saveind in
fragment_X = list()
fragment_y = list()

#defines start frame
start_at = 1
#defines how many frame should be run though until the script ends
for_that_many_frames = 3000
#setting up stuff
bases = ['AgonisticGal3.MPG','M2U00692.MPG','M2U00441.MPG','M2U00442.MPG','wt.mp4','KO.mp4']
#"bases" defines all video which should be analzed
bases = ['test.avi']
#pre = '../Data/Videos/'
pre = ''
#for each video in bases
for base in bases:
  #set up video reader "cap"
  path_vid = pre+base
  print(path_vid)
  print('hold space to analze the video')
  path_background = pre+base[:-4]+'_background.png'
  cap = cv2.VideoCapture(path_vid)
  
  #init background background model
  background = cv2.imread(path_background,2)

  #go to start frame
  for i in range(start_at):
    ret, frame = cap.read()

  #get max frame number
  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  #determine how many frame can be analyzed
  it = min(for_that_many_frames,video_length-start_at)

  #for all frames which are determined to be analyzed
  for i in tqdm(range(it)):
    #get current frame
    ret, frame = cap.read()
    #if there are no more frames
    if(not ret):
      print('video ended')
      break


    #preprocess the current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #subtract background model from current frame
    mask1 = cv2.subtract(gray,background)
    mask2 = cv2.subtract(background,gray)

    #create  a binary image from subtraction
    threshold3 = cv2.inRange(mask2,15,255)
    mask2 = cv2.inRange(mask2,15,255)

    #find all contour for all objects in the foreground
    cnts,_ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts,_ = cv2.findContours(out_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each contour smaller than 3000 pixels, remove that contour
    for cnt in cnts:
      if cv2.contourArea(cnt) < 3000:
        cv2.drawContours(mask2, [cnt], -1, 0, -1)

    #for each contour in the range of 1900 to 60000 pixel
    cnts,_ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    update = np.zeros(gray.shape[:2], np.uint8)
    for cnt in cnts:
      if cv2.contourArea(cnt) > 1900 and cv2.contourArea(cnt) < 60000:
        
        #find smallest rectangle around current contour
        rect = cv2.boundingRect(cnt)
        
        #padd this rectangle by 30
        padding = 30
        boundray_first = max(rect[1]-padding,0)
        boundray_second = min(rect[1]+rect[3]+padding,frame.shape[0])
        boundray_thrid = max(rect[0]-padding,0)
        boundray_fourth = min(rect[0]+rect[2]+padding,frame.shape[1])

        #select regions of the foreground which contain object of the righ size
        data_training_image = frame[boundray_first:boundray_second,boundray_thrid:boundray_fourth]
        data_training_label = mask2[boundray_first:boundray_second,boundray_thrid:boundray_fourth]

        #format these regions into image of the right size and add them to the propper stoarage variable
        data_training_label = cv2.resize(data_training_label, fragment_shape, interpolation = cv2.INTER_AREA)
        data_training_label = np.clip(data_training_label,0,1)*255
        fragment_y.append(data_training_label)
        fragment_X.append(cv2.resize(data_training_image, fragment_shape, interpolation = cv2.INTER_AREA))
        #draw shape in white onto "update" which will be used to update the background model
        cv2.drawContours(update, [cnt], -1, 255, -1)

    #save the information which will be updated into a separete variable (update conatins all the information which are in the foreground)
    prediction = update.copy()
    prepred = update.copy()

    #format prediction and update accordingly
    prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE,np.ones((50,50),np.uint8))
    update = morph(update, cv2.MORPH_DILATE, 50)

    #selcet the infromation which will be used to update the background model from the current frame in "back_up"
    back_up = cv2.bitwise_and(gray,cv2.bitwise_not(update))
    #add back_up to the background 
    background = cv2.bitwise_and(background,update)
    background = cv2.bitwise_or(back_up,background)
    #add  every 50th image and prediction to the right storage variable
    if i % 5 == 0:
      whole_X.append(cv2.resize(frame, whole_shape, interpolation = cv2.INTER_AREA))
      whole_y.append(cv2.resize(prediction, whole_shape, interpolation = cv2.INTER_AREA))

      cv2.imshow('background',background)
      cv2.imshow('frame',frame)
      cv2.imshow('binary',mask2)
      #cv2.imshow('whole2',update)

      if lucidDreaming():
        break

#save training sets
print('whole size:',len(whole_X))
print('fragment size:',len(fragment_X))
np.save('fragment_X.npy',fragment_X)
np.save('fragment_y.npy',fragment_y)
np.save('whole_X.npy',whole_X)
np.save('whole_y.npy',whole_y)

#release and close reader and windows
print('All images were written')
cap.release()
cv2.destroyAllWindows()






