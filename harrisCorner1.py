import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk,Image
import binarization as bin
from skimage import data, img_as_float
imsi = np.array(Image.open("1_1_1.jpg")).astype(np.float32)
height = imsi.shape[0]
width = imsi.shape[1]
x = np.arange(width)
y = np.arange(height)
x, y = np.meshgrid(x, y)
sn=3
sn1=(sn-1)//2
imsi1 = np.zeros([height + sn-1, width + sn-1])
imsi1[y + sn1, x + sn1] = imsi[y, x]
for i in range(sn1):
    imsi1[:,i] = imsi1[:,sn1]
    imsi1[:, -(i+1)] = imsi1[:, -(sn1+1)]
for i in range(sn1):
    imsi1[i] = imsi1[sn1]
    imsi1[-(i+1)] = imsi1[-(sn1+1)]
a=np.zeros([imsi1.shape[0],1])
b=np.zeros([1,imsi1.shape[1]])
Ix=imsi1[:,1:]-imsi1[:,:-1]
Ix=np.hstack((a,Ix))
k=0.04
Iy=imsi1[1:]-imsi1[:-1]
Iy=np.vstack((b,Iy))
imsi2=np.zeros_like(imsi)
for x in range(sn1,imsi1.shape[1]-(sn1+1)):
    for y in range(sn1,imsi1.shape[0]-(sn1+1)):
        M=np.array([[np.sum((Ix[y,x-sn1:x+sn1+1])**2),np.sum((Ix[y,x-sn1:x+sn1+1]*Iy[y-sn1:y+sn1+1,x]))],[np.sum((Ix[y,x-sn1:x+sn1+1]*Iy[y-sn1:y+sn1+1,x])),np.sum((Iy[y-sn1:y+sn1+1,x])**2)]])
        R=np.linalg.det(M)-k*(np.trace(M)**2)
        imsi2[y-sn1,x-sn1]=R
imsi2_xy=np.where(imsi2>imsi2.max()*(1/50))
plt.imshow(imsi,'gray')
plt.plot(imsi2_xy[1],imsi2_xy[0],'r+')
plt.show()

