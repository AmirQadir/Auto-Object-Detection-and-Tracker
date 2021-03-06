#!/usr/bin/python
#
# Copyright 2018 BIG VISION LLC ALL RIGHTS RESERVED
# 
from __future__ import print_function
import sys
import cv2
from random import randint
import argparse
import numpy as np
import cv2 as cv
from yolo_utils import infer_image
#Amir
from skimage.measure import compare_ssim
from skimage.transform import resize
import os
import glob 
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import pickle #for label naming

#from FaceID import faceID


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker



def yolo():
  parser = argparse.ArgumentParser()

  parser.add_argument('-m', '--model-path',
    type=str,
    default='./yolov3-coco/',
    help='The directory where the model weights and \
        configuration files are.')

  parser.add_argument('-w', '--weights',
    type=str,
    default='./yolov3-coco/yolov3.weights',
    help='Path to the file which contains the weights \
        for YOLOv3.')

  parser.add_argument('-cfg', '--config',
    type=str,
    default='./yolov3-coco/yolov3.cfg',
    help='Path to the configuration file for the YOLOv3 model.')

  parser.add_argument('-vo', '--video-output-path',
    type=str,
        default='./output.avi',
    help='The path of the output video file')

  parser.add_argument('-l', '--labels',
    type=str,
    default='./yolov3-coco/coco-labels',
    help='Path to the file having the \
          labels in a new-line seperated way.')

  parser.add_argument('-c', '--confidence',
    type=float,
    default=0.5,
    help='The model will reject boundaries which has a \
        probabiity less than the confidence value. \
        default: 0.5')

  parser.add_argument('-th', '--threshold',
    type=float,
    default=0.3,
    help='The threshold to use when applying the \
        Non-Max Suppresion')

  parser.add_argument('--download-model',
    type=bool,
    default=False,
    help='Set to True, if the model weights and configurations \
        are not present on your local machine.')

  parser.add_argument('-t', '--show-time',
    type=bool,
    default=False,
    help='Show the time taken to infer each image.')

  FLAGS, unparsed = parser.parse_known_args()
  #print(FLAGS)

  # Get the labels
  labels = open(FLAGS.labels).read().strip().split('\n')

  # Intializing colors to represent each label uniquely
  colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

  # Load the weights and configutation to form the pretrained YOLOv3 model
  net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

  # Get the output layer names of the model
  layer_names = net.getLayerNames()
  layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


  ################################



  height, width = frame.shape[:2]

  img, bboxes, _, classid, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

  boxes = [] #It's a list now

  j=0
  for i in classid:
    if i==0:
      print("persons bounding box is: ",bboxes[j])
      boxes.append(bboxes[j].copy())
      #print(boxes[i])
    j=j+1

  print(boxes) #all faces

  ############################temp ###########33
  #for index,value in enumerate(boxes):
  global itr
  for i in range(len(boxes)):
    itr = itr + 1
    # Matching part
    labels = {}






    #Matching part end


    y = boxes[i][1]
    x = boxes[i][0]
    h = boxes[i][3]
    w = boxes[i][2]
    crop_img = img[y:y+h,x:x+w]
    #cv.imwrite(name,crop_img)

    detector = MTCNN()
    print("I am a detector phewww !")
    print(detector.detect_faces(crop_img))
    face_cropped = detector.detect_faces(crop_img)
    if(len(face_cropped)>0):

      boxes_face = (face_cropped[0]['box'])
      y1 = boxes_face[1]
      x1 = boxes_face[0]
      h1 = boxes_face[3]
      w1 = boxes_face[2]
      crop_img_2 = crop_img[y1:y1+h1, x1:x1+w1]
      name = 'dataset/' + str("face")+ str(itr) + '.jpg'
      #cv.imwrite(name,crop_img_2)


    #Matching Part
    path = 'dataset/Face' +str(itr)
    os.mkdir(path)
    name = 'dataset/Face' +str(itr) + '/' + str("person") + str(itr) + ".jpg"
    cv.imwrite(name,crop_img)
    name = 'dataset/Face' +str(itr) + '/' + str("face") + str(itr) + ".jpg"
    try:
      cv.imwrite(name,crop_img_2)
    except:
      pass
    #amtching

    #Faces_Train.py

    os.system('faces_train.py')


    #Faces_Train.py

    with open("labels.pickle",'rb') as f:
      og_labels = pickle.load(f)
      labels = {v:k for k,v in og_labels.items()} #talk to me about this ;) (key, value pair conversion)
    face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml') #only front of a face
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")



    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #scaleFactor aghe peeche krke dekho
    print("Faces",faces)

    for (x, y, w, h) in faces:
      print(x, y, w, h)
      print("Amir")
      roi_gray = gray[y:y+h, x:x+w] #roi -> Region of Interest
      roi_color = crop_img[y:y+h, x:x+w]
      #Recognition part
      id_, conf = recognizer.predict(roi_gray) #id's and confidence level
      print(id_,conf,"moz")
      if(conf>=25 and conf<=85):
        print(id_)
        print("he")
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255,255,255)
        stroke = 2
        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)









  ##########################temp done#########33  

  my_tuple = []

  for i in bboxes:
    my_tuple.append(tuple(i))

  #print(my_tuple)


  # Create MultiTracker object
  multiTracker = cv2.MultiTracker_create()



  # Initialize MultiTracker 
  colors_multi = []
  for bbox in my_tuple:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    colors_multi.append((randint(64, 255), randint(64, 255), randint(64, 255)))

  return multiTracker, colors_multi



if __name__ == '__main__':

  print("Default tracking algoritm is CSRT \n"
        "Available tracking algorithms are:\n")
  for t in trackerTypes:
      print(t)      

  #trackerType = "CSRT"      
  trackerType = "CSRT"      
  itr = 6

  # Set video to load
  videoPath = "muaaz.mp4"
  
  # Create a video capture object to read videos
  cap = cv2.VideoCapture(0)
 
  # Read first frame
  success, frame = cap.read()
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)

  ## Select boxes
  bboxes = []
  colors = [] 
  boxes=[]


  ################# copied code

  multiTracker, colors_multi = yolo()
  





  # Process video and track objects
  




  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break
    
    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, colors_multi[i], 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)
    

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
      break

    if cv2.waitKey(1) & 0xFF == 121:
      multiTracker, colors_multi = yolo()
      print("key presses")  
