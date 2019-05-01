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

if __name__ == '__main__':

  print("Default tracking algoritm is CSRT \n"
        "Available tracking algorithms are:\n")
  for t in trackerTypes:
      print(t)      

  trackerType = "CSRT"      

  # Set video to load
  videoPath = "race.mp4"
  
  # Create a video capture object to read videos
  cap = cv2.VideoCapture(videoPath)
 
  # Read first frame
  success, frame = cap.read()
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)

  ## Select boxes
  bboxes = []
  colors = [] 


  ################# copied code
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

  img, bboxes, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

  my_tuple = []

  for i in bboxes:
    my_tuple.append(tuple(i))

  print(my_tuple)


  # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
  # So we will call this function in a loop till we are done selecting all objects
  #while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
   
    #bbox = cv2.selectROI('MultiTracker', frame)
    #bboxes.append(bbox)
    #colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    #print("Press q to quit selecting boxes and start tracking")
    #print("Press any other key to select next object")
    #k = cv2.waitKey(0) & 0xFF
    #if (k == 113):  # q is pressed
    #  break
  
  #print('Selected bounding boxes {}'.format(bboxes))

  ## Initialize MultiTracker
  # There are two ways you can initialize multitracker
  # 1. tracker = cv2.MultiTracker("CSRT")
  # All the trackers added to this multitracker
  # will use CSRT algorithm as default
  # 2. tracker = cv2.MultiTracker()
  # No default algorithm specified

  # Initialize MultiTracker with tracking algo
  # Specify tracker type
  
  # Create MultiTracker object
  multiTracker = cv2.MultiTracker_create()



  # Initialize MultiTracker 
  colors_multi = []
  for bbox in my_tuple:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    colors_multi.append((randint(64, 255), randint(64, 255), randint(64, 255)))
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

