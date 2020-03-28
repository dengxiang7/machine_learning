from mxnet.gluon import data as gdata
import numpy as np
import struct   #处理数据流
from matplotlib import pyplot as plt
import time
import pandas as pd

def readallimgandwritedata(filename,filename2):
    fp=open(filename, 'rb')
    bin_data=fp.read()
    fp.close()
    data=  bin_data.split()
    data=np.array(data)
    data1=[]
    for i in data:
        if np.float(i) != -1.0:
            data1.append(np.float(i))
    
    
    data1=(np.array(data1)).reshape(int(len(data1)/257) ,257)
   
    
    label=[]
    img=np.empty((int((data1.shape)[0]) ,16, 16))
    for i in range((data1.shape)[0]):
        b=data1[i,:]
        label.append(int(b[0]))
        img[i]=(b[1:257]).reshape((16,16))
        
    fp=open(filename2, 'w')
    for i in range((data1.shape)[0]):
        b=data1[i,:]
        fp.write(str(int(b[0])))
        fp.write("   ")
        for j in range(len(b)):
            if j != 0:
                fp.write(str(j)+':'+str(b[j]))
                fp.write("   ")
        fp.write('\n')
    fp.close()
    return label,img


def showimg(imgs,labs):
    plt.figure(figsize=(12,12))

    for i in range(10):
        plt.subplot(1, 10, i+1,)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(labs[i])
        plt.axis('off')
    plt.show()

def showimg2(imgs,labs):
    for k in range(5):
        showing3(imgs,labs,k)
        
def showing3(imgs,labs,k):
    plt.figure(figsize=(12,12))
    num1=0
    num2=0
    for i in range(len(labs)):  
        if labs[i] == k and num1<10:
            num1+=1
            plt.subplot(1, 20, num1)
            plt.imshow(imgs[i], cmap='gray')
            plt.title(labs[i])
            plt.axis('off')
        plt.subplot(1, 20, 10)
        if labs[i] == k+5 and num2<10:
            num2+=1
            plt.subplot(1, 20, num2+10)
            plt.imshow(imgs[i], cmap='gray')
            plt.title(labs[i])
            plt.axis('off')
    plt.show()

