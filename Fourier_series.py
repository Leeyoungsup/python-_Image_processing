import numpy as np
import math
import matplotlib.pyplot as plt
dx=0.025
x=np.arange(dx,1.25+dx,dx)
y1=np.ones([25])*5
y2=np.zeros([25])*5
y=np.append(y1,y2)
L=1.25/2
plt.plot(x,y)
plt.show()
a0=1/1.25*np.sum(y*dx)
f0=np.ones([50])*a0
plt.plot(x,f0,'red')
plt.show()
result = f0;
for n in range(1,10):
    a=1/L*np.sum(y*np.cos(n*math.pi*x/L)*dx)
    b=1/L*np.sum(y*np.sin(n*math.pi*x/L)*dx)
    result=result+a*np.cos(math.pi*n/L*x)+b*np.sin(math.pi*n/L*x)
    plt.plot(x,y)
    plt.plot(x,result,'red')
    plt.show()

