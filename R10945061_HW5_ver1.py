#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("lena.bmp")


def dilation(img):
    dil = np.zeros((img.shape),np.uint8)
    kernel = np.zeros((5,5),np.uint8)
    kernel[0,1:4]=1
    kernel[1,:]=1
    kernel[2,:]=1
    kernel[3,:]=1
    kernel[4,1:4]=1
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_value = 0
            if (img[i][j]>0).any():
                for row in range(kernel.shape[0]):
                    for column in range(kernel.shape[1]):
                        if i + row < 0:
                            continue
                        elif i + row > (img.shape[0]-1):
                            continue
                        elif j + column <0:
                            continue
                        elif j + column >(img.shape[1]-1):
                            continue
                        else:
                            if (img[i+row][column+j] > max_value).any():
                                max_value = img[i + row][j + column]
                                dil[i][j] = max_value
             
    
    return dil

def erosion(img):
    ero = np.zeros((img.shape),np.uint8)
    kernel = np.zeros((5,5),np.uint8)
    kernel[0,1:4]=1
    kernel[1,:]=1
    kernel[2,:]=1
    kernel[3,:]=1
    kernel[4,1:4]=1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            min_value = 255
            if (img[i][j]>0).any():
                white = True
                for row in range(kernel.shape[0]):
                    for column in range(kernel.shape[1]):
                        if i + row < 0:
                            continue
                        elif i + row > (img.shape[0]-1):
                            continue
                        elif j + column <0:
                            continue
                        elif j + column >(img.shape[1]-1):
                            continue
                        else:
                            if i + row >=0 and i + row <=(img.shape[0]):
                                if (img[i+row][j+column]==0).any():
                                    white = False
                                    break
                                if (img[i+row][j+column]<min_value).any():
                                    min_value = img[i+row][j+column]
                if white ==True:
                    ero[i][j]=min_value
    return ero 
# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_dil = dilation(img)
img_ero = erosion(img)
img_open = dilation(img_ero)
img_close = erosion(img_dil)
# img_dil = cv.cvtColor(img_dil)
# img_ero = erosion(img)

title = ["original","dilation","erosion","opening","closing"]
img = [img,img_dil,img_ero,img_open,img_close]
for i in range(5):
    plt.title(title[i],fontsize =20),plt.imshow(img[i]),plt.figure(i)
plt.show()


# In[ ]:





# In[ ]:




