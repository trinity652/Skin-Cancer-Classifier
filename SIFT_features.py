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
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Hessian Threshold is set to 400
#Paper needs to be read to understand
surf = cv2.xfeatures2d.SIFT_create()

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(gray,None)
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)