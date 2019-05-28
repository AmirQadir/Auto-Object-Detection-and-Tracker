# Auto Object Detection and Tracker

## About
This repository is the result of a semster project we did in CS495 (Fundamentals of Computer Vision) at FAST-NUCES Karachi Pakistan. 

## Objective:
The program is designed to detect various objects from a video stream (either webcam or pre recorded video). The detected objects are then tracked as they move to different positions on the screen. Additionally, the program can identify if the same person has left and then reentered the scene using face recognition.

## Methodology:
The whole program can be divided into the following steps:
Use YOLO V3 to detect objects
Filter out the faces from the detected objects using MTCNN
Save the faces in memory
Use CSRT to track the objects

The above steps run once when the program is executed. Due to limitations of the CSRT tracker, the bounding boxes often lose ‘focus’ (for different reasons such as the object has left the scene). In order to tackle this a key can be pressed to execute the following steps

Use YOLO V3 to detect objects
Filter out the faces from the detected objects using MTCNN
Save the faces in memory
Match the newly detected faces with the previously existing ones using HAAR Cascade
Use CSRT to track the objects

## How to Run
The code was tested on Windows 8 and Windows 10, howver it should work on other versions as well as long as the prerequisites are met. We have mentioned the versions of the libraries/tools we ran this on, again, it may or may not work on other versions.

* Download this Project 
  git clone https://github.com/AmirQadir/Auto-Object-Detection-and-Tracker
* Make sure you have pre-requisites installed.
* python project.py (To run the project).

 

### Prerequisites
* Python 3.7
* OpenCV 3.4.2.16. **Use pip install opencv-contrib-python==3.4.2.16** (The current latest version (4.1.0.25) did not work for us)  
* Imutils **pip install imutils** ( version at time of development: pip install imutils==0.5.2 )
* Numpy **pip install numpy** (version at time of development: pip install numpy==1.16.3 )
* Weights for YOLO https://pjreddie.com/media/files/yolov3.weights

## References

YOLO V3
https://pjreddie.com/darknet/yolo/

CSRT (And other Trackers) 
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/









