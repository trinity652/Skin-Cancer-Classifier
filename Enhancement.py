#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:31:42 2017

@author: abhilasha
"""
import system
import cv2
import numpy as np
from matplotlib import pyplot as plt

class segmentationKmeans:
    def __init__(self,path):
        self.pathToImages= path
        self.files = os.listdir(path)
    #Segmentation based on k-means clustering; to be specific color quantization
    def kmeanTest(self):#The object needs to be passed to the methods everytime
        i=1
        for file in self.files:
            path=self.pathToImages+'/' + str(i)+'.jpg'
            img = cv2.imread(path)
            img=img.reshape((-1,3))
            img = np.float32(img)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 15
            ret,label,center=cv2.kmeans(img,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            


        