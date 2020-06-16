import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
import math
def determinant(b,mat1):
    k=mat1[0].size
    value=0
    if k==2:
        value=value+((mat1[0,0]*mat1[1,1])-(mat1[1,0]*mat1[0,1]))
        return value
    else:
        for j in range(k):
            c=b*((-1)**(2+j))*mat1[0,j]
            mat2=np.delete(mat1,j,axis=1)
            mat2=np.delete(mat2,0,axis=0)
            if k==3:
                value=value+(c*((mat2[0,0]*mat2[1,1])-(mat2[1,0]*mat2[0,1])))
            else:
                value=value+determinant(c,mat2)
        return value
def myInv(ad,d):
    mat3=np.dot(d,ad)
    Aabs=mat3[0,0]
    value=ad/Aabs
    return value
im1 = pilimg.open('I8_1.png')
imsi1= np.array(im1)
imsi1=imsi1[:,:,2]
im2 = pilimg.open('I8_2.png')
imsi2= np.array(im2)
imsi2=imsi2[:,:,2]
x=np.array(imsi1[1,:].shape)
y=np.array(imsi1[:,1].shape)
win=np.zeros((800,600))
py=np.arange(y)
px=np.arange(x)
[px,py]=np.meshgrid(px,py)
a=np.zeros((30,2,1))
ado=np.zeros((30,2,1))
b=np.zeros((30,4,1))
bdo=np.zeros((30,4,1))
h=np.zeros((4,4))
hdo=np.zeros((4,4))
x1=np.array([ 66., 180., 295., 414., 529.,  68., 183., 300., 425., 533.,  66.,\
       187., 304., 425., 540.,  66., 187., 308., 432., 546.,  68., 193.,\
       312., 438., 555.,  68., 191., 317., 438., 564.])
y1=np.array([ 52.,  54.,  50.,  52.,  48., 182., 182., 182., 182., 165., 319.,\
       319., 314., 312., 308., 453., 453., 451., 448., 448., 596., 596.,\
       587., 587., 585., 741., 741., 738., 730., 730.])
x2=np.array([ 53., 170., 271., 401., 514.,  40., 163., 276., 401., 518.,  38.,\
       167., 287., 414., 533.,  40., 167., 297., 425., 548.,  40., 180.,\
       308., 432., 546.,  53., 185., 306., 438., 557.])
y2=np.array([ 41.,  48.,  52.,  50.,  41., 195., 197., 197., 191., 165., 340.,\
       338., 334., 327., 310., 477., 470., 461., 455., 451., 613., 600.,\
       591., 593., 600., 749., 745., 756., 754., 747.])
for i in range(30):
    a[i]=np.array(([x2[i]],[y2[i]]))
    b[i]=np.array(([x1[i]*y1[i]],[x1[i]],[y1[i]],[1]))
    ado[i]=np.array(([x1[i]],[y1[i]]))
    bdo[i]=np.array(([x2[i]*y2[i]],[x2[i]],[y2[i]],[1]))
for i in range(5):
    for j in range(4):
        c=np.concatenate((a[i*5+j],a[i*5+j+1],a[i*5+j+5],a[i*5+j+6]), axis=1)
        d=np.concatenate((b[i*5+j],b[i*5+j+1],b[i*5+j+5],b[i*5+j+6]), axis=1)
        cdo=np.concatenate((ado[i*5+j],ado[i*5+j+1],ado[i*5+j+5],ado[i*5+j+6]), axis=1)
        ddo=np.concatenate((bdo[i*5+j],bdo[i*5+j+1],bdo[i*5+j+5],bdo[i*5+j+6]), axis=1)
        for k in range(4):
            for g in range(4):
                mat2=np.delete(d,g,axis=1)
                mat2=np.delete(mat2,k,axis=0)
                h[k,g]=((-1)**(2+k+g))*determinant(1,mat2)
                mat3=np.delete(ddo,g,axis=1)
                mat3=np.delete(mat3,k,axis=0)
                hdo[k,g]=((-1)**(2+k+g))*determinant(1,mat3)
        ad=np.transpose(h)
        addo=np.transpose(hdo)
        mata=myInv(ad,d)
        matado=myInv(addo,ddo)
        ab=np.dot(c,mata)
        abdo=np.dot(cdo,matado)
        w1=((b[i*5+j,2]-b[i*5+j+1,2])/(b[i*5+j,1]-b[i*5+j+1,1]))*(px-b[i*5+j+1,1])+b[i*5+j+1,2]
        w2=((b[i*5+j+5,2]-b[i*5+j+6,2])/(b[i*5+j+5,1]-b[i*5+j+6,1]))*(px-b[i*5+j+5,1])+b[i*5+j+5,2]
        w3=((b[i*5+j+1,1]-b[i*5+j+6,1])/(b[i*5+j+1,2]-b[i*5+j+6,2]))*(py-b[i*5+j+1,2])+b[i*5+j+1,1]
        w4=((b[i*5+j+5,1]-b[i*5+j,1])/(b[i*5+j+5,2]-b[i*5+j,2]))*(py-b[i*5+j+5,2])+b[i*5+j+5,1]
        jeh=np.array(np.where((py<=w2)&(py>=w1)&(px<=w3)&(px>=w4)))
        jeh1=np.array([jeh[0]*jeh[1],jeh[1],jeh[0],1])
        jeh2=np.dot(ab,jeh1)
        jeh2x=jeh2[1].astype(np.int)
        jeh2y=jeh2[0].astype(np.int)
        win[jeh2x,jeh2y]=imsi1[jeh[0],jeh[1]]
        w1=((bdo[i*5+j,2]-bdo[i*5+j+1,2])/(bdo[i*5+j,1]-bdo[i*5+j+1,1]))*(px-bdo[i*5+j+1,1])+bdo[i*5+j+1,2]
        w2=((bdo[i*5+j+5,2]-bdo[i*5+j+6,2])/(bdo[i*5+j+5,1]-bdo[i*5+j+6,1]))*(px-bdo[i*5+j+5,1])+bdo[i*5+j+5,2]
        w3=((bdo[i*5+j+1,1]-bdo[i*5+j+6,1])/(bdo[i*5+j+1,2]-bdo[i*5+j+6,2]))*(py-bdo[i*5+j+1,2])+bdo[i*5+j+1,1]
        w4=((bdo[i*5+j+5,1]-bdo[i*5+j,1])/(bdo[i*5+j+5,2]-bdo[i*5+j,2]))*(py-bdo[i*5+j+5,2])+bdo[i*5+j+5,1]
        jeh3=np.array(np.where((py<=w2)&(py>=w1)&(px<=w3)&(px>=w4)))
        jehq=np.array(np.where(win[jeh3[0],jeh3[1]]==0))
        jeh3=np.array(jeh3[:,jehq])
        jeh3do=np.array([jeh3[0]*jeh3[1],jeh3[1],jeh3[0],1])
        jeh4=np.dot(abdo,jeh3do)
        jeh4x=jeh4[1].astype(np.int)
        jeh4y=jeh4[0].astype(np.int)
        win[jeh3[0],jeh3[1]]=imsi1[jeh4x,jeh4y]
        plt.imshow(win,'gray')
plt.show()
