#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 00:02:42 2017

@author: abhilasha
"""
"""
samples : It should be of np.float32 data type, and each feature should be put in a single column.

nclusters(K) : Number of clusters required at end

criteria : It is the iteration termination criteria. When this criteria is satisfied, 
algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are ( type, max_iter, epsilon ):
3.a - type of termination criteria : It has 3 flags as below:
cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached. cv2.TERM_CRITERIA_MAX_ITER - 
stop the algorithm after the specified number of iterations, max_iter. cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - 
stop the iteration when any of the above condition is met.

3.b - max_iter - An integer specifying maximum number of iterations.

3.c - epsilon - Required accuracy

attempts : Flag to specify the number of times the algorithm is executed using different initial labellings. 
The algorithm returns the labels that yield the best compactness. This compactness is returned as output.

flags : This flag is used to specify how initial centers are taken. Normally two flags are used for this : 
cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS.

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./Melanoma/12.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8;
"""
compactness : It is the sum of squared distance from each point to their corresponding centers.
labels : This is the label array (same as ‘code’ in previous article) where each element marked ‘0’, ‘1’.....
centers : This is array of centers of clusters.
"""
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
print(ret)
print(type(label))
print(label)
print(center)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.imshow(res2,cmap = 'gray')