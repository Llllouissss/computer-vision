#!/usr/bin/env python
# coding: utf-8

# In[95]:


#(a) a histogram
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("lena.bmp")


plt.hist(img.ravel(),256,(0,256),color="black")
plt.figure(0)
plt.imshow(img)
plt.figure(1)
plt.xlabel("pixel brightness")
plt.ylabel("pixel amounts")
plt.show()

print(img.size)


# In[96]:


#(b) image with intensity divided by 3 and its histogram
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("lena.bmp")


plt.hist(img.ravel()/3,256,(0,256),color="black")
plt.xlabel("pixel brightness")
plt.ylabel("pixel amounts")
plt.show()
plt.figure(0)
img_divided = (img//3).astype(np.int8)
plt.imshow((img_divided))


# In[94]:


#(c) image after applying histogram equalization to (b) and its histogram
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("lena.bmp",cv.COLOR_BGR2GRAY)

img_divided = (img//3)

a = []
c = 0
b = []
for i in np.unique(img_divided):
    x = np.sum(img_divided==i)
    a.append(x)
    x = x/ img_divided.size
# 求出個別機率
    c = c +x
# 累加機率
    g = round(c*255)
#  乘以像素值得到均衡化的數值(並四捨五入)
    b.append(g)
plt.bar(b,a,color = "black")
plt.xlabel("pixel brightness")
plt.ylabel("pixel amounts")
plt.figure(0)
img_eq = cv.equalizeHist(img_divided)

img_eq = np.expand_dims(img_eq,axis = 2)

img_eq = cv.cvtColor(img_eq,0)

plt.imshow(img_eq)
plt.figure(1)
plt.show()


    


# In[ ]:





# In[ ]:




