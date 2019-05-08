from FaceID import faceID
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



img1 = cv.imread('nabeel.jpg',0)          # queryImage
img2 = cv.imread('nabeel_train.jpg',0) # trainImage

print(img1.shape)
rec = faceID()
print("constructor finished")
# crop_img_2 = getCroppedImage(rec,crop_img_2) accepts image in np arary

print(img1.shape)
img1 = cv.resize(img1,(100,100),interpolation=cv.INTER_AREA)
print(img1.shape)
img1 = rec.prewhiten2(img1)

print(img1.shape)
# print("whiten finished")
embeds = rec.getEmbed(img1) 
# print("embedding finished")
