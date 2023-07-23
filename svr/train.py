# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 13:49:40 2023

@author: Thomas Skøtt Gummesen
"""


import os
import skimage

#The atomicfile module, which is used to save segment files, is a publically available python file i had no part in making
from atomicfile import AtomicFile

from sklearn.svm import SVR
from skimage.segmentation import slic
from skimage.io import imread
from skimage.util import img_as_float
import joblib
import numpy as np
import errno
import math
import time
import cv2
#These constance can be changed to vary the output of the algorithm. they have not been optimized for this dataset

N_SEGMENTS=40
segment_size=math.ceil(150/(math.sqrt(N_SEGMENTS)))
trainfull=0         #If this is set to 1, it will train on all of the data, which will take several hours to run, 
#assuming SVR is scales quadratically and it is the only contributer to computation time, since it took 1 hour for 20% it should take 8 hours for 80% on my laptop

            

def eachimage(path):
    ##This part segments the coloured image
    img=img_as_float(imread(path))
    img2 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
    segments=slic(img2,n_segments=N_SEGMENTS,compactness=10,sigma=1,channel_axis=None)
    segments=np.stack((segments,segments,segments),axis=2)
    
    #Blurring the images after segmentation, as the SLIC algorithm already has a gaussian kernel
    img2=cv2.GaussianBlur(src=img2,ksize=(3,3), sigmaX=1,sigmaY=1)
    img=cv2.GaussianBlur(src=img,ksize=(3,3), sigmaX=1,sigmaY=1)
    


    #The number of segments, since first segment is 0, we add 1
    #I had several indexing issues that took a lot of trial and error
    #So a bunch of times i added 1 or subtracted 1 to a value or a loop
    n_segments=segments.max()+1
    #number of segments does not match N_SEGMENTS, but will not be higher.

    #Converts the image from 0-1 RGB colorspace into LAB colorspace with smaller scale
    img4=np.round(img*255).astype(np.uint8)
    lab=skimage.color.rgb2lab(img4)
    lab[:,:,0]=lab[:,:,0]/100  #sets the scale from 0-1
    lab[:,:,1]=lab[:,:,1]/128   #sets the scale from -1-1
    lab[:,:,2]=lab[:,:,2]/128   #sets the scale from -1-1
    #I found it easier to deal with that range than what LAB uses, it is convertet back to standard ranes for LAB at the very end of test script

    ## Construct the center of each of the segments found through SLIC clustering
    segment_points=np.zeros(n_segments)
    center=np.zeros((n_segments,2))
    luminance=np.zeros(n_segments)        #Luminance of each centroid
    A=np.zeros(n_segments)
    B=np.zeros(n_segments)
    for (i,j), k in np.ndenumerate(segments[:,:,0]):
        segment_points[k]=segment_points[k]+1   #Total number of pixels in each segment
        center[k][0]=center[k][0]+i
        center[k][1]=center[k][1]+j
        luminance[k]=luminance[k]+img[i][j][0]    #Sum of luminance over whole image
        A[k]=A[k]+lab[i][j][1]
        B[k]=B[k]+lab[i][j][2]
    #I often in my code had issues with indexing, am used to start counting at 1, so sometime i end up add 1 or subtracting 1 to loops to get them to fucktion as i want
    for k in range(n_segments):   
        if k!=0:
            center[k]=center[k]/segment_points[k]   #This gives us the center of each segment
            A[k]=A[k]/segment_points[k]             #This gives a simple mean of the a value of each segment
            B[k]=B[k]/segment_points[k]             #This gives a simple mean of the b value of each segment
            luminance[k]=luminance[k]/segment_points[k] #This gives a simple mean of the l value of each segment
    center=center[1:,:] #Segmenting issues
    luminance=luminance[1:]#Segmenting issues
    A=A[1:]#Segmenting issues
    B=B[1:]#Segmenting issues
    
    ##Here we do a fast fourier transformation (FFT2) around the centroid of each segment
    subsquares=np.zeros((n_segments-1,segment_size*segment_size))   #Segmenting issues
    for k in range(n_segments-1):   #Segmenting issues
        edge1=int(center[k][0])  #Finds location of center across axis 0
        if img.shape[0]<=edge1+segment_size:
            edge1=img.shape[0]-1-segment_size #If center is too close to bottom
        if edge1-segment_size<=0:
            edge1=segment_size+1                #If center is too close to top
        edge2=int(center[k][1])  #Finds location of center across 1 axis
        if img.shape[1]<=edge2+segment_size:
            edge2=img.shape[1]-1-segment_size #If center is too close to right edge
        if edge2-segment_size<=0:
            edge2=segment_size+1                #If center is too close to top
        for i in range(segment_size):
            for j in range(segment_size):
                centering=int(segment_size/2)
                      #Identifying the four edges was not to much trouble
                subsquares[k][i*segment_size+j]=lab[i-centering+edge1][j-centering+edge2][0]    #centering is to ensure fourier transformation is done around center of segment
            #Finding the correct way to shape the subsquares i was not able to do correctly, i used a modified version of what i found in, taken from Harrison and Varun
        subsquares[k]=np.fft.fft2(subsquares[k].reshape(segment_size,segment_size)).reshape(segment_size*segment_size) #I struggled a lot getting the correct dimensionality of the 2DFFT
    return subsquares,A,B



start=time.time()
folder=r'C:\Users\Thomas\Documents\MATLAB\Computational Imaging\project\color'
if trainfull==1:
    trainset=np.load('trainset.npy')
else:
    trainset=np.load('subtrainset.npy')
#The "trainset.npy" can be used to specify how much of the data is trained on
X=np.array([]).reshape(0,segment_size*segment_size)
Atrain=np.array([])
Btrain=np.array([])
j=sum(trainset)
n=round(100*j/np.shape(trainset)[0])
print('Training on',n,'% of',np.shape(trainset)[0],'Images in LAB color space')
i=0
k=np.zeros(j)
for root,dirs,files in os.walk(folder): # These file management scripts i dont really understand, so the os.walk functions are, taken from Harrison and Varun
    for file in files: 
        path=os.path.join(root,file)#These file path lines are taken from Harrison and Varun
        if not path.endswith(".jpg"):
            continue
        file2=int(file.removesuffix('.jpg'))#This one excludes stuf with jpg in folder, taken from Harrison and Varun
        if trainset[file2]==1:
            if i>0:
                k[i]=math.ceil(i/j*100) #I use this so the script periodically gives feedback on performance
                if k[i]>k[i-1]:
                    print(int(k[i]),'% of subsquaress generated')
            i=i+1
            subsquares,A,B=eachimage(path)      #This is the part that call the big function defined in start
            X=np.concatenate((X,subsquares),axis=0) #I was not able to get the correct dimensions here so like in the eachimage function, i refered to, Harrison and Varun
            Atrain=np.concatenate((Atrain,A),axis=0)
            Btrain=np.concatenate((Btrain,B),axis=0)
            
epsilon=0.08 #For the values of the constant, i looked at which , Harrison and Varun used and chose something in that range and tried a few different values in that range
C=0.1     #For the values of the constant, i looked at which , Harrison and Varun used and chose something in that range and tried a few different values in that range
print('Fitting model for A')
a_svr=SVR(C=C,epsilon=epsilon)
a_svr.fit(X,Atrain)
joblib.dump(a_svr,f'models/a_{n}%_n={N_SEGMENTS}svr.model')
#After this model is dumpe you have what you need to test for the a term in the test script if you were to rewrite it a bit
print('Fitting model for B')
b_svr=SVR(C=C,epsilon=epsilon)
b_svr.fit(X,Btrain)
joblib.dump(b_svr,f'models/b_{n}%_n={N_SEGMENTS}svr.model')
print('Script finished')
end=time.time()
print(round(end-start),'seconds runtime for training on',round(100*j/np.shape(trainset)[0]),'% of images')
    #I wanted to compare running times between the different sizes of training sets