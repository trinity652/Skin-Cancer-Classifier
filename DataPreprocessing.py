#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:48:49 2017

@author: abhilasha
"""
#Renaming the files in the directories
import os
from PIL import Image

class readFiles:
    # Class to read the path and rename the images sequentially
    def __init__(self,path):
        self.pathToImages= path
        self.files = os.listdir(path)
    def rename(self):
        i=1
        for file in self.files:
            os.rename(os.path.join(self.pathToImages, file), os.path.join(self.pathToImages, str(i)+'.jpg'))
            i = i+1
    def rgbToGray(self):
        i=1
        for file in self.files:
            path=self.pathToImages+'/' + str(i)+'.jpg'
            #img=Image.open(path).convert('L')
            img=Image.open(path).resize((1200,1000),Image.ANTIALIAS)
            img.save(path)
            i=i+1



Folder1 = readFiles("./Melanoma")
Folder1.rename()
Folder1.rgbToGray()

Folder2 = readFiles("./Basal Cell Carcinoma")
Folder2.rename()
Folder2.rgbToGray()

Folder3 = readFiles("./Squamous Cell Carcinoma")
Folder3.rename()
Folder3.rgbToGray()








    





    