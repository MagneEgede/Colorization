# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:06:59 2023

@author: Thomas
"""
import time
import skimage
from skimage.segmentation import slic
from skimage.io import imread
import sklearn
from skimage.util import img_as_float
import joblib
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

#This script runs slow and gives loads of warnings when it runs. but it runs.
N_SEGMENTS=40   #If you wanna change number of segments you gotta change it in th train_final.npy and run that first
segment_size=math.ceil(150/(math.sqrt(N_SEGMENTS)))
plotting=False
image_nr=19
##if plotting is true, then it will plot image_nr, if plotting is false it will do error analysis on number of images specified by trainset_final.py

def metrics_compare(a_svr,b_svr,a_max,b_max,o):
    #quite professional of me to keep my python scripts in my matlab folder
    
    path=('C:\\Users\\Thomas\\Documents\\MATLAB\\Computational Imaging\\project\\color\\'+str(o)+'.jpg')    #This is the path for RGB images
    path2=('C:\\Users\\Thomas\\Documents\\MATLAB\\Computational Imaging\\project\\gray\\'+str(o)+'.jpg')    #This is the path for coloured images
    img=img_as_float(imread(path))
    img2=img_as_float(imread(path2))
    
    
    ##This part segments the image
    ##This segment is copy-paste from trainset_final.npy
    segments=slic(img,n_segments=N_SEGMENTS,compactness=10,sigma=1)
    segments=np.stack((segments,segments,segments),axis=2)
    
    #Blurring the images after segmentation, as the SLIC algorithm already has a gaussian kernel
    img2=cv2.GaussianBlur(src=img2,ksize=(3,3), sigmaX=1,sigmaY=1)
    img=cv2.GaussianBlur(src=img,ksize=(3,3), sigmaX=1,sigmaY=1)
    
    
    
    #The number of segments, since first segment is 0, we add 1
    #I had several indexing issues that took a lot of trial and error
    #So a bunch of times i added 1 or subtracted 1 to a value or a loop
    n_segments=segments.max()+1
    #number of segments does not match N_SEGMENTS, but will not be higher.
    

    ##This segment is largely the same as what is in trainset_final.npy, few differences 
    ## Construct the center of each of the segments found through SLIC clustering
    segment_points=np.zeros(n_segments)
    center=np.zeros((n_segments,2))
    luminance=np.zeros(n_segments)        #Luminance of each centroid
    for (i,j), k in np.ndenumerate(segments[:,:,0]):
        segment_points[k]=segment_points[k]+1   #Total number of pixels in each segment
        center[k][0]=center[k][0]+i
        center[k][1]=center[k][1]+j
        luminance[k]=luminance[k]+img[i][j][0]    #Sum of luminance over whole image
    #I often in my code had issues with indexing, am used to start counting at 1, so sometime i end up add 1 or subtracting 1 to loops to get them to fucktion as i want
    for k in range(n_segments):   
        if k!=0:
            center[k]=center[k]/segment_points[k]   #This gives us the center of each segment
            luminance[k]=luminance[k]/segment_points[k] #This gives a simple mean of the l value of each segment
    center=center[1:,:] #Segmenting issues
    luminance=luminance[1:]#Segmenting issues
    
    
    ##This segment is copy-pasted from trainset_final
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
                offshot=int(segment_size/2)
                      #Identifying the four edges was not to much trouble
                subsquares[k][i*segment_size+j]=img[i-offshot+edge1][j-offshot+edge2][0]    #offshot is to ensure fourier transformation is done around center of segment
            #Finding the correct way to shape the subsquares i was not able to do correctly, i used a modified version of what i found in, taken from Harrison and Varun
        subsquares[k]=np.fft.fft2(subsquares[k].reshape(segment_size,segment_size)).reshape(segment_size*segment_size) #I struggled a lot getting the correct dimensionality of the 2DFFT    
    
    
    
    
    
    segments=segments-1 #Another issue i had with indexing
    ##Here we for each segment makes a list for which other segments are adjecent, which is used in MRF
    ##Though MRF did not work, the script i wrote for finding adjacent segments did
    adjacent=np.zeros([n_segments,n_segments])

    barrier=np.zeros(np.shape(img2))    #This array tells me where the boundaries between different segments are
    for (i,j), current in np.ndenumerate(segments[:,:,0]):
        if i<img2.shape[0]-1:   #avoid checking on the edges of the image
            nearby=segments[i+1][j][0]
            if current!=nearby:
                barrier[i][j]=1     #This gives a binary array showing where the boundaries between segments are
                adjacent[current][nearby]=adjacent[current][nearby]+1
                adjacent[nearby][current]=adjacent[nearby][current]+1
        if j<img2.shape[1]-1:   #avoid checking on the edges of the image
            nearby=segments[i][j+1][0]
            if current!=nearby:
                barrier[i,j]=1      #This gives a binary array showing where the boundaries between segments are
                adjacent[current][nearby]=adjacent[current][nearby]+1
                adjacent[nearby][current]=adjacent[nearby][current]+1
    #This was used in my MRF code that i could not get to work
    adjacent=np.minimum(adjacent,np.ones([n_segments,n_segments]))
    #It is a logical array telling identifying which segment each segment is adjacent to
    
    
    #There used to be a segment here about my MRF, but i could not get it to work
    
    
    
   
    


    ##Here we use the model that was trained to predict A and B using SVR. We limit its outputs to the limits of A and B after they are scaled down
    a_guess=np.maximum(np.minimum(a_svr.predict(subsquares)*2,a_max),-a_max)
    b_guess=np.maximum(np.minimum(b_svr.predict(subsquares)*2,b_max),-b_max)
    
    
    ## I had a messed up code for trying to do MRF, if i applied its output it made the output images nonsense, so i removed it

    ## This part reconstructs the image rom the predicted L, A and B from greyscale and the model
    lab=np.zeros(np.shape(img)) #LAB is our guess for coloured image
    img=img_as_float(imread(path))
    lab2=skimage.color.rgb2lab(np.round(img*255).astype(np.uint8))  #This is the LAB value of the true color image
    lab[:,:,0]=lab2[:,:,0]/100  #sets the scale from 0-1
    

    for (i,j), value in np.ndenumerate(segments[:,:,0]):
       lab[i][j][1]=a_guess[value-1]  #Subtract 1 again, indexing issues
       lab[i][j][2]=b_guess[value-1]  #Subtract 1 again, indexing issues
       
    
    ##I had scaled my lab down to range -1 to 1, not i scale back up to what lab is supposed to be
    lab[:,:,0]=lab[:,:,0]*100  #Converts back to proper LAB
    lab[:,:,1]=lab[:,:,1]*128  #Converts back to proper LAB
    lab[:,:,2]=lab[:,:,2]*128  #Converts back to proper LAB
    #As mentioned in the report, the L term does not match the greyscale
    rgb=skimage.color.lab2rgb(lab)

       
    ##This is the segment for computing the errors
    lab2=lab2
    lab=lab
    #Here lab2 is is the original color image in LAB form
    #Here lab is the trained color image in LAB form

    
    color_lab=lab2
    train_lab=lab
    color_rgb=np.round(skimage.color.lab2rgb(color_lab)*255).astype(np.uint8)
    train_rgb=np.round(skimage.color.lab2rgb(train_lab)*255).astype(np.uint8)
    a_color=color_lab[:,:,1]
    a_train=train_lab[:,:,1]
    b_color=color_lab[:,:,2]
    b_train=train_lab[:,:,2]
    MAE1=np.zeros(2).T
    RMSE1=np.zeros(2).T
    MAE1[0]=sklearn.metrics.mean_absolute_error(a_color,a_train)
    MAE1[1]=sklearn.metrics.mean_absolute_error(b_color,b_train)
    RMSE1[0]=sklearn.metrics.mean_squared_error(a_color,a_train, squared=False)
    RMSE1[1]=sklearn.metrics.mean_squared_error(b_color,b_train, squared=False)
    SSIM1=skimage.metrics.structural_similarity(color_rgb,train_rgb,data_range=train_rgb.max()-train_rgb.min(),channel_axis=2)
    PSNR1=20*math.log10(255.0/math.sqrt(np.mean((color_rgb-train_rgb)**2)))
    return MAE1,RMSE1,SSIM1,PSNR1,rgb,img,img2,barrier


start=time.time()

trainset=np.load('subtrainset.npy')
j=sum(trainset)
n=round(100*j/np.shape(trainset)[0])
a_svr=joblib.load(f'models/a_{n}%_n={N_SEGMENTS}svr.model')
b_svr=joblib.load(f'models/b_{n}%_n={N_SEGMENTS}svr.model')

trainset=np.load('trainset.npy')
j=sum(trainset)



a_max=1
b_max=1

testset=(trainset-1)*-1
testset=testset.astype(int)
MAE=np.zeros((sum(testset),2))
RMSE=np.zeros((sum(testset),2))
SSIM=np.zeros((sum(testset)))
PSNR=np.zeros((sum(testset)))
if plotting==True:
    MAE1,RMSE1,SSIM1,PSNR1,testimage,color,gray,barrier=metrics_compare(a_svr,b_svr,a_max,b_max,image_nr)
    withbarrier=np.stack(((barrier-1)*-1,(barrier-1)*-1,(barrier-1)*-1),axis=2)
    withbarrier1=(barrier-1)*-1
    if plotting:
       fig=plt.figure(frameon=False)
       ax=plt.Axes(fig,[0.,0.,1.,1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(color)
       plt.show()
    if plotting:
       fig=plt.figure(frameon=False)
       ax=plt.Axes(fig,[0.,0.,1.,1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(gray, cmap='gray')
       plt.show()
    if plotting:
       fig=plt.figure(frameon=False)
       ax=plt.Axes(fig,[0.,0.,1.,1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(gray*withbarrier1, cmap='gray')
       plt.show()
    if plotting:
       fig=plt.figure(frameon=False)
       ax=plt.Axes(fig,[0.,0.,1.,1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(testimage)
       plt.show()   
    if plotting:
       fig=plt.figure(frameon=False)
       ax=plt.Axes(fig,[0.,0.,1.,1.])
       ax.set_axis_off()
       fig.add_axes(ax)
       ax.imshow(testimage*withbarrier)
       plt.show()    
    end=time.time()
    end=time.time()
    print(round(end-start),'seconds runtime for testing on plotting stuff for image',image_nr)
else:
    o=0
    j=sum(testset)
    k=np.zeros(j)
    print('Evaluating errors on a testset of',j,'images')
    for i in range(np.shape(trainset)[0]):
        if testset[i]==1:
            if o>0:
                k[o]=math.ceil(o/j*100) #I use this so the script periodically gives feedback on performance
                if k[o]>k[o-1]:
                    print(int(k[o]),'% of subsquaress generated')
            MAE1,RMSE1,SSIM1,PSNR1,testimage,color,gray,barrier=metrics_compare(a_svr,b_svr,a_max,b_max,i)
            MAE[o,:]=MAE1
            RMSE[o,:]=RMSE1
            SSIM[o]=SSIM1
            PSNR[o]=PSNR1
            o=o+1
    MAE_a_std=np.std(MAE[:,0])
    MAE_a_value=np.mean(MAE[:,0])
    MAE_b_std=np.std(MAE[:,1])
    MAE_b_value=np.mean(MAE[:,1])
    RMSE_a_std=np.std(RMSE[:,0])
    RMSE_a_value=np.mean(RMSE[:,0])
    RMSE_b_std=np.std(RMSE[:,1])
    RMSE_b_value=np.mean(RMSE[:,1])
    SSIM_std=np.std(SSIM)
    SSIM_mean=np.mean(SSIM)
    PSNR_std=np.std(PSNR)
    PSNR_mean=np.mean(PSNR)
    end=time.time()
    print(round(end-start),'seconds runtime for testing on',o,'images')
