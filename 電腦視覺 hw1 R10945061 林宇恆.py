#!/usr/bin/env python
# coding: utf-8

# In[80]:


#(a) upside-down lena.bmp
import numpy as np 

import cv2 as cv

import matplotlib.pyplot as plt


img = cv.imread("lena.bmp")
def flipp(img):
    img2 = np.zeros([512,512,3],np.uint8)
    for i in range(512):
        img2[i,:]=img[512-i-1,:]
    return img2

x = flipp(img)

plt.imshow(x)
plt.axis('off')


# In[81]:


#(b) right-side-left lena.bmp
import numpy as np 

import cv2 as cv

import matplotlib.pyplot as plt


img = cv.imread("lena.bmp")
def flipp(img):
    img2 = np.zeros([512,512,3],np.uint16)
    for i in range(512):
        img2[:,i]=img[:,512-i-1]
    return img2

x = flipp(img)

plt.imshow(x)
plt.axis('off')


# In[82]:


#(c) diagonally flip lena.bmp
import numpy as np 

import cv2 as cv

import matplotlib.pyplot as plt


img = cv.imread("lena.bmp")
def flipp(img):
    img2 = np.zeros([512,512,3],np.uint8)
    for i in range(512):
        for j in range(512):
            img2[512-i-1,j]=img[i,512-j-1]
    return img2

x = flipp(img)

plt.imshow(x)
plt.axis('off')


# In[83]:


#(d) rotate lena.bmp 45 degrees clockwise
import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

img = cv.imread("lena.bmp")

def rotate(img):
    (h1,w1,d1) = img.shape
    center = (h1//2,w1//2)
    R = cv.getRotationMatrix2D(center,315,0.7)
    img2 = cv.warpAffine(img,R,(512,512))
    
    return img2   
x = rotate(img)
plt.imshow(x)
plt.axis('off')


# In[51]:


#(e) shrink lena.bmp in half
import cv2 as cv

import numpy as np

import matplotlib.pyplot as ply

img = cv.imread("lena.bmp")

imgresize = cv.resize(img,(256,256))

plt.imshow(imgresize)
plt.axis('off')


# In[79]:


#(f) binarize lena.bmp at 128 to get a binary image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("lena.bmp",0)

ret,img2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

image = [img2]

for i in range(1):
    plt.imshow(img2,'gray')
    
plt.axis('off')


# In[ ]:





# In[ ]:





# In[ ]:




