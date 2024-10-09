

from mxnet.gluon import data as gdata
import numpy as np
import struct   #处理数据流
import sys
from matplotlib import pyplot as plt
import time

# '../../img/train-images.idx3-ubyte'  '../../img/train-labels.idx1-ubyte'

def readallimg(filename):
    fp=open(filename, 'rb')
    bin_data=fp.read()
    fp.close()
    
    offset = 0
    fmt_header = '>iiii' # 以大端法读取4个 unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header) #计算给定的格式(fmt)占用多少字节的内存，注意对齐方式
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols)) #注意empty创建的数组中，包含的均是无意义的数值。
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images
  
def readalllabels(flename):
    fp=open(flename, 'rb')
    bin_data=fp.read()
    fp.close()
    
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
                  

def showimg(imgs,labs):
    plt.figure(figsize=(12,12))

    for i in range(10):
        plt.subplot(1, 10, i+1,)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(labs[i])
        plt.axis('off')
    plt.show()


def take01img(filename1,filename2):
    x=readallimg(filename1)
    y=readalllabels(filename2)
    t_x=[]
    t_y=[]
    for i in range(len(y)):
        if y[i]==0:
            t_x.append(x[i])
            t_y.append(y[i])
        if y[i]==1:
            t_x.append(x[i])
            t_y.append(y[i])
    return t_x ,t_y

f="../img/train-images.idx3-ubyte"
ff="../img/train-labels.idx1-ubyte"
