import numpy as np
import random
from numpy.linalg import matrix_rank
n=random.randrange(5,6)
a=np.zeros((n,n))
def matrix():
    for i in range(n):
        for j in range(n):
            a[i,j]=random.randrange(0,5)
    if matrix_rank(a)!=n:
        raise ValueError
    return a
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
    mat3=np.dot(mat,ad)
    Aabs=mat3[0,0]
    value=ad/Aabs
    
    return value
while(1):
    try:
        mat=matrix()
    except ValueError:
        print('행렬을 새로 설정합니다')
    else:
        break
matrixvalue=0
M=np.zeros((n,n))
ad=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        mat2=np.delete(mat,j,axis=1)
        mat2=np.delete(mat2,i,axis=0)
        M[i,j]=((-1)**(2+i+j))*determinant(1,mat2)
ad=np.transpose(M)
mata=myInv()
