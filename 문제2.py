import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import math
pi=3.14
hand= np.array(pilimg.open('_7_5_1.jpg'))
edge=np.array(np.where(hand[:,:,1]>150))
wcx=(np.max(edge[0])+np.min(edge[0]))/2
wcy=(np.max(edge[1])+np.min(edge[1]))/2
a=int(edge.size//2)
th=np.zeros((a))
d=np.zeros((a))
do=np.zeros((360))
for i in range(a):
    d[i]=math.sqrt((edge[0,i]-wcx)**2+(edge[1,i]-wcy)**2)
    th[i]=math.atan2(-edge[1,i]+wcy,-edge[0,i]+wcx)*180/pi+180
    q=int(round(th[i]))
    if do[q]<=d[i]:
        do[q]=d[i]
for s in range(1,359):
    if do[s]==0:
        g,h=s+1,s-1
        while(1):
            if h!=0 and do[h]==do[h-1]:
                h=h-1
            else:
                break
        while(1):
            if g!=359 and do[g]==do[g+1]:    
                g=g+1
            else:
                break
        if h!=0 and g!=359:
            do[s]=((do[h]-do[g])/(h-g))*(s-h)+do[h]
for s in range(1,359):
    do[s]=(do[s-1]+do[s+1])/2
the=np.arange(int(np.min(th)),int(np.max(th)))
plt.plot(the,do[int(np.min(th)):int(np.max(th))],'b-')
plt.show()
