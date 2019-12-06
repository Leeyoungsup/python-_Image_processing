import PIL.Image as pilimg
import numpy as np  # numpy임포트
import matplotlib.pyplot as plt
from scipy.fftpack import fft2,ifft2
import time
t=time.time()
def Fourier_transform(image,M,N):
    F=np.copy(image)
    pi=np.pi
    x=np.arange(M)
    y=np.arange(N)
    x,y=np.meshgrid(x,y)
    value1=np.exp(-2j*pi*x/M)
    value2=np.exp(-2j*pi*y/N)
    CT=np.zeros([xs,xs,ys])
    CT=CT.astype(np.complex128)
    ST=np.zeros([ys,ys,xs])
    ST=ST.astype(np.complex128)
    imsift=image.astype(np.complex128)
    for u in range(M):
        CT[u]=value1**u*F
    for v in range(N):
        ST[v]=value2**v
    for i in range(M):
            for j in range(N):
                    imsift[i,j]=np.sum(CT[j]*ST[i])
    return imsift
def Inverse_Fourier_transform(image,M,N,u,v):
    f=np.copy(image)
    pi=np.pi
    x=np.arange(M)
    y=np.arange(N)
    x,y=np.meshgrid(x,y)
    value=f*(np.cos(2*pi*(u*x/M+v*y/N))+1j*np.sin(2*pi*((u*x/M)+(v*y/N))))
    valuef=np.sum(value)/(M*N)
    print(valuef)
    return valuef
imsi = pilimg.open('1_3_1.jpg')
imsi=np.array(imsi)
imsi1=np.copy(imsi)
imsi_ft=np.copy(imsi)
imsi_ft=imsi_ft.astype(np.complex128)
ys=imsi.shape[0]
xs=imsi.shape[1]
imsi_ft=Fourier_transform(imsi,xs,ys)
print(time.time()-t)
plt.imshow(20*np.log(np.abs(np.fft.fftshift(imsi_ft))),'gray')
plt.figure()
plt.imshow(20*np.log(np.abs(np.fft.fftshift(fft2(imsi1)))),'gray')
plt.show()
