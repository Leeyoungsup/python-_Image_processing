import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import math
im = pilimg.open('I.png')
imsi= np.array(im)
hb=1558021
hb=(hb%100)*2
th =math.radians(hb)
sin=math.sin(th)
cos=math.cos(th)
x=np.array(imsi[1,:,1].shape)
y=np.array(imsi[:,1,1].shape)
wi=np.zeros((y[0],x[0]))
wi=wi+255
w=np.array(([cos,-sin],[sin,cos]))
ps=np.array(([0,x,0,x],[0,0,y,y]))
w=w*3
rot=np.dot(w,ps)
rot1=np.copy(rot)
rot1[0]=rot1[0]-np.min(rot1[0])
wx=int(np.max(rot[0])-np.min(rot[0]))
wy=int(np.max(rot[1]))
win=np.zeros((wy,wx))
py=np.arange(y)
px=np.arange(x)
ws=np.array(([cos,-sin,-np.min(rot[0])],[sin,cos,0],[0,0,1]))
[px,py]=np.meshgrid(px,py)
jeh=np.array(np.where(win==0))
pxy=np.array((px,py,1))
pxy1=np.dot(ws,pxy)
pxy2=np.dot(ws,pxy)
pxy2[0]=pxy1[0].astype(np.int32)
pxy2[1]=pxy1[1].astype(np.int32)
win[pxy2[1]-1,pxy2[0]-1]=np.array(imsi[y-pxy[1]-1,pxy[0]-1,2])
jeh=np.array(np.where(win==0))
w1=np.array(np.where(jeh[0]==0))
w2=np.array(np.where(jeh[1]==0))
w3=np.array(np.where(jeh[0]==np.max(jeh[0])))
w4=np.array(np.where(jeh[1]==np.max(jeh[1])))
w=np.concatenate((w1,w2,w3,w4),axis=1)
jeh=np.delete(jeh,w,axis=1)
a=np.array(np.where(np.logical_or(win[jeh[0]+1,jeh[1]]!=0,win[jeh[0]-1,jeh[1]]!=0)))
win[jeh[0,a],jeh[1,a]]=win[jeh[0,a]+1,jeh[1,a]]
plt.imshow(win,'gray',origin='lower')
plt.show()
