#!/usr/bin/env python
# coding: utf-8

# In[28]:


#(a) a binary image (threshold at 128)
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("lena.bmp")
ret,img2 = cv.threshold(img,128,255,cv.THRESH_BINARY)
plt.imshow(img2)
plt.show()


# In[26]:


#(b) a histogram
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("lena.bmp",0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])

plt.bar(range(1,257), hist.ravel(),color="black")
plt.xlabel("pixel brightness")
plt.ylabel("pixel percentage")
plt.show()



# In[27]:


#(b) a histogram
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("lena.bmp",0)

plt.hist(img.ravel(),256,(0,256),color="black")
plt.xlabel("pixel brightness")
plt.ylabel("pixel percentage")
plt.show()


# In[ ]:





# In[57]:


#(c) connected components
(regions with + at centroid, 
bounding box)
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.bmp",cv2.COLOR_BGR2GRAY)
ret,img3 = cv2.threshold(img, 128 ,255 ,cv2.THRESH_BINARY)
img3 = np.expand_dims(img3, axis=2)
plt.figure(0)
# plt.imshow(img3)
num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(img3, connectivity=8, ltype=None)

# output = np.zeros((img3.shape[0],img3.shape[1],3),np.uint8)
# for i in range(1,num_labels):
#     mask = labels == i
#     output[:,:,0][mask] = np.random.randint(0,255)
#     output[:,:,1][mask] = np.random.randint(0,255)
#     output[:,:,2][mask] = np.random.randint(0,255)
# 
# output = np.zeros((img3.shape[0],img3.shape[1],1),np.uint8)
# mask = labels == 1
# output[:,:][mask] = np.random.randint(0,255)
# plt.figure(1)
# plt.imshow(output)

# print(stats[1])
# print(type(stats[1][0])) #x
# print(stats[1][1].shape) #y
# print(stats[1][2].shape) #w
# print(stats[1][3].shape) #h
# print(stats[1][4].shape) #area

img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
print(img3.shape)

print(f"centroids[0] = {centroids[0]}, type = {type(centroids[0])}")

x = stats[1][0]
y = stats[1][1]
w = stats[1][2]
h = stats[1][3]
area = stats[1][4]

for i in range(1,len(stats)):
    if stats[i][4] >= 500:
        x = stats[i][0]
        y = stats[i][1]
        w = stats[i][2]
        h = stats[i][3]
        center_pt = (int(centroids[i][0]), int(centroids[i][1]))
        cv2.rectangle(img3,(x,y),(x+w,y+h),(0,0,255),5)
        cv2.circle(img3, center_pt, radius=5, color=(255, 0, 0), thickness=-1)

print(f"x = {x}, type = {type(x)}")
print(f"y = {y}, type = {type(y)}")
print(f"x = {w}, type = {type(w)}")
print(f"x = {h}, type = {type(h)}")
print(f"x = {area}, type = {type(area)}")



plt.figure(2)
plt.imshow(img3)


# In[ ]:




