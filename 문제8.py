import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
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
def myInv():
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
b=np.zeros((30,4,1))
h=np.zeros((4,4))
x1=np.array([ 68, 178, 297, 421, 529,  68, 180, 297, 421, 533,  68,\
       187, 300, 427, 540,  68, 187, 306, 432, 544,  64, 189,\
       312., 436, 555,  66, 189, 315, 440, 564])
y1=np.array([ 52,  54,  54,  54,  50, 180, 184, 182, 182, 165, 314,\
       319, 312, 310, 312, 457, 451, 451, 453, 446, 596, 596,\
       589, 589, 589, 745, 741, 738, 734, 728])
x2=np.array([ 48, 170, 271, 397, 516,  38, 163, 276, 403, 518,  44,\
       167, 284, 416, 535,  44, 167, 300, 425, 548,  40, 180,\
       310., 434, 548,  55, 185, 306, 438, 561])
y2=np.array([ 41,  44,  50,  48,  37, 197, 193, 199, 189, 165, 340,\
       338, 336, 329, 310, 474, 472, 464, 451, 453, 609, 593,\
       593, 593, 602, 751, 745, 751, 754, 751])

for i in range(30):
    a[i]=np.array(([x2[i]],[y2[i]]))
    b[i]=np.array(([x1[i]*y1[i]],[x1[i]],[y1[i]],[1]))
for i in range(5):
    for j in range(4):
        c=np.concatenate((a[i*5+j],a[i*5+j+1],a[i*5+j+5],a[i*5+j+6]), axis=1)
        d=np.concatenate((b[i*5+j],b[i*5+j+1],b[i*5+j+5],b[i*5+j+6]), axis=1)
        for k in range(4):
            for g in range(4):
                mat2=np.delete(d,g,axis=1)
                mat2=np.delete(mat2,k,axis=0)
                h[k,g]=((-1)**(2+k+g))*determinant(1,mat2)
        ad=np.transpose(h)
        mata=myInv()
        ab=np.dot(c,mata)
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
        plt.imshow(win,'gray')
plt.show()
        



