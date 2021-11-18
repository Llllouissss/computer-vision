#!/usr/bin/env python
# coding: utf-8

# In[60]:


import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("lena.bmp")
img_comp = np.zeros(img.shape,np.uint8)
for i in range(0,len(img.ravel())):
    img_comp.ravel()[i] = 255 - img.ravel()[i]
    

def binary(img):
    for i in range(0,len(img.ravel())):
        if img.ravel()[i] > 127:
            img.ravel()[i] = 255
        elif img.ravel()[i]<=127:
            img.ravel()[i] = 0
    return img

def dilation(img):
    dil = np.zeros(img.shape,np.uint8)
    kernel = np.uint8(np.zeros((5,5)))
    kernel[0,1:-1]=1
    kernel[1,:]=1
    kernel[2,:]=1
    kernel[3,:]=1
    kernel[4,1:-1]=1
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (img[y][x] > 0).all():
                for row in range(kernel.shape[0]):
                    for column in range(kernel.shape[1]):
                        if y +row < 0:
                            continue
                        elif y +row >(img.shape[0]-1):
                            continue
                        elif x +column <0:
                            continue
                        elif x +column >(img.shape[1]-1):
                            continue
                        else:
                            dil[y+row][x+column] = 255
    return dil    

def erode(img):
    ero = np.zeros(img.shape,np.uint8)
    kernel = np.uint8(np.zeros((5,5)))
    kernel[0,1:-1]=1
    kernel[1,:]=1
    kernel[2,:]=1
    kernel[3,:]=1
    kernel[4,1:-1]=1
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (img[y][x] > 0 ).all():
                white = True
                for row in range(kernel.shape[0]):
                        for column in range(kernel.shape[1]): 
                            if y +row <0:
                                white = False
                            elif y +row >(img.shape[0]-1):
                                white = False
                            elif x +column <0:
                                white = False
                            elif x +column >(img.shape[1]-1):
                                white = False
                            elif (img[y+row][x+column] == 0).any():
                                white = False
                if white:
                    ero[y][x] = 255
                                

    return ero    

def hit_and_miss(img):
    j_kernel = np.uint8(np.zeros((2,2)))
    k_kernel = np.uint8(np.zeros((2,2)))
    j_kernel[0,:]=1
    j_kernel[1,0]=1
    k_kernel[0,1]=1
    k_kernel[1,:]=1
    
# EROSION with img  
    ero_1 = np.zeros(img.shape, np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (img[y][x] > 0).any():
                white = True
                for row in range(j_kernel.shape[0]):
                    for column in range(j_kernel.shape[1]):
                        if y+row < 0:
                            white = False
                        elif y+row > (img.shape[0]-1):
                            white = False
                        elif x+column < 0:
                            white = False
                        elif x+column > (img.shape[1]-1):
                            white = False
                        elif (img[y+row][x+column] == 0).any():
                            white = False
                if white:
                    ero_1[y][x] = 255
#     return ero_1

# EROSION with img_comp

   
    ero_2 = np.zeros(img.shape, np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (img_comp[y][x] > 0).any():
                white = True
                for row in range(k_kernel.shape[0]):
                    for column in range(k_kernel.shape[1]):
                        if y+row < 0:
                            white = False
                        elif y+row > (img.shape[0]-1):
                            white = False
                        elif x+column < 0:
                            white = False
                        elif x+column > (img.shape[1]-1):
                            white = False
                        elif (img_comp[y+row][x+column] == 0).any():
                            white = False
                if white:
                    ero_2[y][x] = 255
#     return ero_2
    #compare ero and ero_2
    
    hit = np.zeros(img.shape, np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
#             if (ero_2[y][x] != 255).any():
            if(ero_1[y][x] == ero_2[y][x]).any():
                hit[y][x] = 255
    return hit



img_bi = binary(img)

img_dil = dilation(img_bi)
img_ero = erode(img_bi)
img_open = dilation(img_ero)
img_close = erode(img_dil)

img_bi_comp = binary(img_comp)
img_hit = hit_and_miss(img_bi)
img_hit1 = erode(img_hit)
title = ["original","dilation","erosion","opening","closing","hit and miss"]
img = [img,img_dil,img_ero,img_open,img_close,img_hit]

for i in range(6):
    plt.title(title[i]),plt.imshow(img[i]),plt.figure(i)
    
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




