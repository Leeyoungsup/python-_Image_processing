import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import math
pi=3.14
hand=np.zeros((10,200,200,3))
hand[0]=np.array(pilimg.open('_7_1_1.jpg'))
hand[1]=np.array(pilimg.open('_7_2_1.jpg'))
hand[2]=np.array(pilimg.open('_7_3_1.jpg'))
hand[3]=np.array(pilimg.open('_7_4_1.jpg'))
hand[4]=np.array(pilimg.open('_7_5_1.jpg'))
hand[5]=np.array(pilimg.open('_7_6_1.jpg'))
hand[6]=np.array(pilimg.open('_7_7_1.jpg'))
hand[7]=np.array(pilimg.open('_7_8_1.jpg'))
hand[8]=np.array(pilimg.open('_7_9_1.jpg'))
hand[9]=np.array(pilimg.open('_7_10_1.jpg'))
for i in range(10):
    edge=np.array(np.where(hand[i][:,:,1]>150))
    a=edge.size//2
    wcx=(np.max(edge[0])+np.min(edge[0]))/2
    wcy=(np.max(edge[1])+np.min(edge[1]))/2
    plt.subplot(5,2,i+1)
    plt.plot(wcx,wcy,'r.')
    plt.plot(edge[1],200-edge[0],'r.')
plt.show()
th=np.zeros((a))
d=np.zeros((a))
for i in range(10):
    edge=np.array(np.where(hand[i][:,:,1]>150))
    wcx=(np.max(edge[0])+np.min(edge[0]))/2
    wcy=(np.max(edge[1])+np.min(edge[1]))/2
    d=np.sqrt((edge[0]-wcx)**2+(edge[1]-wcy)**2)
    th=(np.arctan2(-edge[1]+wcy,-edge[0]+wcx))*180/pi+180
    plt.subplot(5,2,i+1)
    plt.plot(th,d,'b.')
plt.show()
