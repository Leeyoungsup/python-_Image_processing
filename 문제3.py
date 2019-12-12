import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import math
im = pilimg.open('I.png')
imsi= np.array(im)
hb=1558021
hb=(hb%100)*2
th = math.radians(hb)
sin=math.sin(th)
cos=math.cos(th)
x=np.array(imsi[1,:,1].shape)
y=np.array(imsi[:,1,1].shape)
w=np.array(([cos,-sin],[sin,cos]))
ps=np.array(([0,x,0,x],[0,0,y,y]))
rot=np.dot(w,ps)
wx=int(np.max(rot[0])-np.min(rot[0]))
wy=int(np.max(rot[1]))
win=np.zeros((wy,wx))
py=np.arange(y)
px=np.arange(x)
ws=np.array(([cos,-sin,-np.min(rot[0])],[sin,cos,0],[0,0,1]))
[px,py]=np.meshgrid(px,py)
pxy=np.array((px,py,1))
pxy1=np.dot(ws,pxy)
pxy2=np.dot(ws,pxy)
pxy2[0]=pxy1[0].astype(np.int32)
pxy2[1]=pxy1[1].astype(np.int32)
win[pxy2[1]-1,pxy2[0]-1]=np.array(imsi[y-pxy[1]-1,-1+pxy[0],2])
plt.imshow(win,'gray',origin='lower')
plt.show()
