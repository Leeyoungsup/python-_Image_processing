import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter.filedialog as tkfile
from PIL import ImageTk, Image
window = Tk()
filename=tkfile.askopenfilenames(title = "Select file",parent = window, filetypes = (("jpg files","*.jpg"),("all files","*.*")))
pfile=np.array(filename)
pfile_index=pfile.shape[0]
K=10
hr=15
means_Clustering=np.random.rand(K,128)*10

all_SIFT=np.zeros((1,128))
for file in range(pfile_index):
    under_index=pfile[file].find('_')
    under_index1 = pfile[file][under_index+1:].find('_')
    imsi = np.array(Image.open(pfile[file])).astype(np.float32)
    for i in range(1, len(pfile[file])):
        if pfile[file][-i] == '/':
            filename_index = -i
            break

    SIFT=np.load("./SIFT_max("+str(hr)+" 100)_save/"+pfile[file][under_index+1:under_index+1+under_index1]+"/"+pfile[file][filename_index+1:-4]+".npy")
    all_SIFT=np.vstack((all_SIFT,SIFT))
all_SIFT=np.delete(all_SIFT,0,axis=0)
while(1):
    count = np.ones(K)
    temporary_means_Clustering=np.copy(means_Clustering)
    for Clustering in range(all_SIFT.shape[0]):
        means_Clustering_index=np.sqrt(((means_Clustering-all_SIFT[Clustering])**2).sum(axis=1)).argmin()
        count[means_Clustering_index]+=1
        temporary_means_Clustering[means_Clustering_index]+=all_SIFT[Clustering]
    for i in range(K):
        temporary_means_Clustering[i]=temporary_means_Clustering[i]/count[i]
    if (means_Clustering==temporary_means_Clustering).min()==True:
        break
    means_Clustering=np.copy(temporary_means_Clustering)
np.save("./finalSIFT/"+str(K)+" means Clustering max("+str(hr)+" 100).npy",means_Clustering)
insik=np.zeros((10,K))
insik_count=np.zeros((10))
for file in range(pfile_index):
    under_index=pfile[file].find('_')
    under_index1 = pfile[file][under_index+1:].find('_')
    imsi = np.array(Image.open(pfile[file])).astype(np.float32)
    for i in range(1, len(pfile[file])):
        if pfile[file][-i] == '/':
            filename_index = -i
            break
    SIFT=np.load("./SIFT_max("+str(hr)+" 100)_save/"+pfile[file][under_index+1:under_index+1+under_index1]+"/"+pfile[file][filename_index+1:-4]+".npy")
    for SIFT_index in range(SIFT.shape[0]):
        temporary_Kmeans=(means_Clustering - SIFT[SIFT_index])**2
        temporary_Kmeans_index=np.sqrt(temporary_Kmeans.sum(axis=1)).argmin()
        insik[int(pfile[file][under_index+1:under_index+1+under_index1])-1,temporary_Kmeans_index]+=1
    insik_count[int(pfile[file][under_index+1:under_index+1+under_index1])-1]+=1
for i in range(10):
    insik[i]=insik[i]/insik_count[i]
np.save("./finalSIFT/"+str(K)+" means insik max("+str(hr)+" 100).npy",insik)