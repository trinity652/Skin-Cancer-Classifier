#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 03:22:58 2017

@author: abhilasha
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Using SURF for a single image
img = cv2.imread('./Melanoma/9.jpg',0)

#Hessian Threshold is set to 400
#Paper needs to be read to understand
surf = cv2.xfeatures2d.SURF_create(500)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
des = np.array(des,np.float32)
print(kp)
print(des)
print(len(kp))
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()
print (surf.descriptorSize())
