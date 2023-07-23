# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 19:33:28 2023

@author: Thomas
"""
import numpy as np
import random
import math
subtrainsize=20 #This determines how much data you train on if you dont train on the whole set
subtestset=500  #How many number of images will be tested on

nimages=7129
ntrain=math.ceil(7129*0.8)      #We use 80% of the images
ntrain2=math.ceil(7129*subtrainsize/100)     #Sometimes we use less than 80%, in that case we use a subset of the 80%
trainset=np.zeros(shape=(nimages))
for i in range(ntrain):
    trainset[i]=1
random.seed(18072023)
random.shuffle(trainset)
trainset=trainset.astype(int)
trainset2=np.zeros(shape=(sum(trainset)))
for i in range(ntrain2):
    trainset2[i]=1
random.shuffle(trainset2)
trainset3=np.zeros(shape=(nimages))
np.save('trainset.npy',trainset)
j=0
for i in range(nimages):
    if trainset[i]==1:
        if trainset2[j]==1:
            trainset3[i]=1
        j=j+1
trainset3=trainset3.astype(int)
np.save('subtrainset.npy',trainset3)
j=0
testset=np.zeros(shape=(nimages))
for i in range(nimages):
   if trainset[i]==0:
       if j<subtestset:
           testset[i]=1
           j=j+1
np.save('100testset.npy',testset)       #Sometimes i tested on less than the 20% remaining, that is what this one is for