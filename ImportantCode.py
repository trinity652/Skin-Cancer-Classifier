#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:48:49 2017

@author: abhilasha
"""

#Code Snippets important for creating the project
"""
#Renaming the files in the directories
import os
path = '/Users/myName/Desktop/directory'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
    i = i+1

#Converting the images into gray scale

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('image.png')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

#Compression of images

# My image is a 200x374 jpeg that is 102kb large
foo = Image.open("path\\to\\image.jpg")
foo.size
# I downsize the image with an ANTIALIAS filter (gives the highest quality)
foo = foo.resize((160,300),Image.ANTIALIAS)
foo.save("path\\to\\save\\image_scaled.jpg",quality=95)
# The saved downsized image size is 24.8kb
foo.save("path\\to\\save\\image_scaled_opt.jpg",optimize=True,quality=95)
 # The saved downsized image size is 22.9kb

#
 Color Quantization
Color Quantization is the process of reducing number of colors in an image.
 One reason to do so is to reduce the memory. Sometimes, 
 some devices may have limitation such that it can produce only limited number of colors.
 In those cases also, color quantization is performed. Here we use k-means clustering for color quantization.

There is nothing new to be explained here. There are 3 features, say, 
R,G,B. So we need to reshape the image to an array of Mx3 size (M is number of pixels in image).
 And after the clustering, we apply centroid values (it is also R,G,B) to all pixels, such that resulting
 image will have specified number of colors. And again we need to reshape it back to the shape of original image.
 Below is the code:"""

import numpy as np
import cv2

img = cv2.imread('home.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#High pass filter for each of the images and then adding to the original

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('dave.jpg',0)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

#Feature extraction using Histogram of Gradients

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure


image = color.rgb2gray(data.astronaut())

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()

#Feature extraction of the images using Haar Transform

import pywt
import matplotlib.pyplot as plt
import numpy as np

ts = [2, 56, 3, 22, 3, 4, 56, 7, 8, 9, 44, 23, 1, 4, 6, 2]

(ca, cd) = pywt.dwt(ts,'haar')

cat = pywt.thresholding.soft(ca, np.std(ca)/2)
cdt = pywt.thresholding.soft(cd, np.std(cd)/2)

ts_rec = pywt.idwt(cat, cdt, 'haar')

plt.close('all')

plt.subplot(211)
# Original coefficients
plt.plot(ca, '--*b')
plt.plot(cd, '--*r')
# Thresholded coefficients
plt.plot(cat, '--*c')
plt.plot(cdt, '--*m')
plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
plt.grid('on')

plt.subplot(212)
plt.plot(ts)
plt.hold('on')
plt.plot(ts_rec, 'r')
plt.legend(['original signal', 'reconstructed signal'])
plt.grid('on')
plt.show()

#SVM

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  

# Linear Kernel
svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted= svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)

# Output

[[16  0  0]
 [ 0 13  5]
 [ 0  4  7]]





"""