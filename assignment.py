import tkinter
import cv2
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import numpy as np
import matplotlib.pyplot as plt
window=tkinter.Tk()

window.title("1558021_이영섭_영상처리연습")
window.geometry("1024x800")
window.resizable(1, 1)
label=tkinter.Label(window)
sub_img=0
histogram=0
def func_exit():
    window.quit()
    window.destroy()
def func_open():
    global label,sub_img
    fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.tiff"), ("all files", "*.*")) )
    src = cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(src)
    sub_img=np.copy(src)
    imgtk = ImageTk.PhotoImage(image=img)
    label.configure(image=imgtk)
    label.image=imgtk
def func_save():
    fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.jpg"), ("all files", "*.*")) )
    cv2.imwrite(fileName,sub_img)
def func_calcHisto():
    global sub_img,histogram
    histogram=np.zeros((255))
    sum_histogram=np.zeros((255))
    for i in range(0,255):
        histogram[i]=np.count_nonzero(sub_img==[i])
    plt.bar(range(len(histogram)),histogram)
    plt.show()
def func_equalHisto():
    global sub_img,label1
    
    histogram=np.zeros((255))
    sum_histogram=np.zeros((255))
    for i in range(0,255):
        histogram[i]=np.count_nonzero(sub_img==[i])
        sum1= histogram[0:i].sum()
        sum_histogram[i]=sum1
    sum_histogram=sum_histogram*(255/sub_img.size)
    src1=sum_histogram[sub_img]
    src1=src1.astype(np.int32)
    
    toplevel=tkinter.Toplevel(window)
    toplevel.geometry("600x600")
    label=tkinter.Label(toplevel)
    label.pack(expand=True)
    img = Image.fromarray(src1)
    imgtk = ImageTk.PhotoImage(image=img)
    label.configure(image=imgtk)
    label.image=imgtk
    def func_save1():
        fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.jpg"), ("all files", "*.*")) )
        cv2.imwrite(fileName,src1)
    def func_exit1():
        toplevel.quit()
        toplevel.destroy()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="저장",command=func_save1)
    button.pack()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="취소",command=func_exit1)
    button.pack()
def func_GaussF():
    global sub_img
    sn=9
    sn1=sn//2
    sigma=10
    x=np.arange(0,sn)-sn1
    y=np.arange(0,sn)-sn1
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    height = sub_img.shape[0]
    width = sub_img.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1, width + sn1])
    imageValue1 = np.copy(imageValue)
    imageValue[y + sn1, x+sn1] = sub_img[y, x]
    for i in range(sn1, height ):
        for j in range(sn1, width ):
            value = Mask * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            imageValue1[i, j] = np.sum(value)
    imageValue1=(imageValue1/imageValue1.max())*255
    toplevel=tkinter.Toplevel(window)
    toplevel.geometry("600x600")
    label=tkinter.Label(toplevel)
    label.pack(expand=True)
    img = Image.fromarray(imageValue1)
    imgtk = ImageTk.PhotoImage(image=img)
    label.configure(image=imgtk)
    label.image=imgtk
    def func_save1():
        fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.jpg"), ("all files", "*.*")) )
        cv2.imwrite(fileName,imageValue1)
    def func_exit1():
        toplevel.quit()
        toplevel.destroy()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="저장",command=func_save1)
    button.pack()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="취소",command=func_exit1)
    button.pack()   
def func_1stDF():
    global sub_img
    sn=3
    sn1=sn//2
    Mask=np.zeros((3,3))
    Mask[:,0]=-1
    Mask[:,2]=1
    Mask1=np.zeros((3,3))
    Mask1[0,:]=-1
    Mask1[2,:]=1
    height = sub_img.shape[0]
    width = sub_img.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1, width + sn1])
    imageValue1 = np.copy(imageValue)
    imageValue2 = np.copy(imageValue)
    imageValue[y + sn1, x+sn1] = sub_img[y, x]
    for i in range(sn1, height ):
        for j in range(sn1, width ):
            value = Mask * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            value1 = Mask1 * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            imageValue1[i, j] = np.sum(value)
            imageValue2[i, j] = np.sum(value1)
    imageValue1=abs(imageValue1)+abs(imageValue2)
    toplevel=tkinter.Toplevel(window)
    toplevel.geometry("600x600")
    label=tkinter.Label(toplevel)
    label.pack(expand=True)
    img = Image.fromarray(imageValue1)
    imgtk = ImageTk.PhotoImage(image=img)
    label.configure(image=imgtk)
    label.image=imgtk
    def func_save1():
        fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.jpg"), ("all files", "*.*")) )
        cv2.imwrite(fileName,imageValue1)
    def func_exit1():
        toplevel.quit()
        toplevel.destroy()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="저장",command=func_save1)
    button.pack()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="취소",command=func_exit1)
    button.pack()     

def GaussF(sigma):
    global sub_img
    sn=9
    sn1=sn//2
    x=np.arange(0,sn)-sn1
    y=np.arange(0,sn)-sn1
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    height = sub_img.shape[0]
    width = sub_img.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1, width + sn1])
    imageValue1 = np.copy(imageValue)
    imageValue[y + sn1, x+sn1] = sub_img[y, x]
    for i in range(sn1, height ):
        for j in range(sn1, width ):
            value = Mask * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            imageValue1[i, j] = np.sum(value)
    imageValue1=(imageValue1/imageValue1.max())*255
    return imageValue1
def func_dogF():
    global sub_img
    imageValue1=GaussF(5)-GaussF(1)
    
    toplevel=tkinter.Toplevel(window)
    toplevel.geometry("600x600")
    label=tkinter.Label(toplevel)
    label.pack(expand=True)
    img = Image.fromarray(imageValue1)
    imgtk = ImageTk.PhotoImage(image=img)
    label.configure(image=imgtk)
    label.image=imgtk
    def func_save1():
        fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.jpg"), ("all files", "*.*")) )
        cv2.imwrite(fileName,imageValue1)
    def func_exit1():
        toplevel.quit()
        toplevel.destroy()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="저장",command=func_save1)
    button.pack()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="취소",command=func_exit1)
    button.pack()
def Es(image):
    sn=3
    sn1=sn//2
    Mask=np.zeros((3,3))
    Mask[:,0]=-1
    Mask[:,2]=1
    Mask1=np.zeros((3,3))
    Mask1[0,:]=-1
    Mask1[2,:]=1
    height = image.shape[0]
    width = image.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1, width + sn1])
    imageValue1 = np.copy(imageValue)
    imageValue2 = np.copy(imageValue)
    imageValue[y + sn1, x+sn1] = image[y, x]
    for i in range(sn1, height ):
        for j in range(sn1, width ):
            value = Mask * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            value1 = Mask1 * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            imageValue1[i, j] = np.sum(value)
            imageValue2[i, j] = np.sum(value1)
    imageValue1=abs(imageValue1)+abs(imageValue2)
    return imageValue1
def Eo(image):
    sn=3
    sn1=sn//2
    Mask=np.zeros((3,3))
    Mask[:,0]=-1
    Mask[:,2]=1
    Mask1=np.zeros((3,3))
    Mask1[0,:]=-1
    Mask1[2,:]=1
    height = image.shape[0]
    width = image.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1, width + sn1])
    imageValue1 = np.copy(imageValue)
    imageValue2 = np.copy(imageValue)
    imageValue[y + sn1, x+sn1] = image[y, x]
    for i in range(sn1, height ):
        for j in range(sn1, width ):
            value = Mask * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            value1 = Mask1 * imageValue[i - sn1:i +sn1+1, j - sn1:j + sn1+1]
            imageValue1[i, j] = np.sum(value)
            imageValue2[i, j] = np.sum(value1)
    eo=np.degrees(np.arctan2(imageValue2,imageValue1))+180
    return eo
def func_canny():
    global sub_img
    imageValue1=GaussF(1)
    es=Es(imageValue1)
    src1=Eo(imageValue1)
    height = es.shape[0]
    width = es.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    In=np.zeros((height+2,width+2))
    In[y+1,x+1]=es[y,x]
    d1=np.where(((src1>=337.5)|(src1<=22.5))|((src1<=202.5)&(src1>=157.5)),1,0)
    d2=np.where(((src1>22.5)&(src1<=67.5))|((src1<247.5)&(src1>202.5)),1,0)
    d3=np.where(((src1>67.5)&(src1<=112.5))|((src1<292.5)&(src1>=247.5)),1,0)
    d4=np.where(((src1>112.5)&(src1<157.5))|((src1<337.5)&(src1>=292.5)),1,0)
    In1=np.copy(In)
    for i in (np.argwhere(d1==1)+1):
        In1[i[0],i[1]]=np.where(In[i[0],i[1]]==In[i[0]-1:i[0]+2,i[1]-1:i[1]+2].max(),In[i[0],i[1]],0)
    for i in (np.argwhere(d2==1)+1):
        In1[i[0],i[1]]=np.where(In[i[0],i[1]]==In[i[0]-1:i[0]+2,i[1]-1:i[1]+2].max(),In[i[0],i[1]],0)
    for i in (np.argwhere(d3==1)+1):
        In1[i[0],i[1]]=np.where(In[i[0],i[1]]==In[i[0]-1:i[0]+2,i[1]-1:i[1]+2].max(),In[i[0],i[1]],0)
    for i in (np.argwhere(d4==1)+1):
        In1[i[0],i[1]]=np.where(In[i[0],i[1]]==In[i[0]-1:i[0]+2,i[1]-1:i[1]+2].max(),In[i[0],i[1]],0)
    Tk=127
    Ti=30
    In2=np.where(In1>Tk,255,0)
    
    
    for j in range(5):
        InP=np.argwhere(In2==255)
        for i in InP:
            if (d1[i[0]-1,i[1]-1]==1):
                In2[i[0]-1,i[1]]=np.where(In[i[0]-1,i[1]]>=Ti,255,0)
                In2[i[0]+1,i[1]]=np.where(In[i[0]+1,i[1]]>=Ti,255,0)
            elif (d2[i[0]-1,i[1]-1]==1):
                In2[i[0]+1,i[1]-1]=np.where(In[i[0]+1,i[1]-1]>=Ti,255,0)
                In2[i[0]-1,i[1]+1]=np.where(In[i[0]-1,i[1]+1]>=Ti,255,0)

            elif (d3[i[0]-1,i[1]-1]==1):
                In2[i[0],i[1]-1]=np.where(In[i[0],i[1]-1]>=Ti,255,0)
                In2[i[0],i[1]+1]=np.where(In[i[0],i[1]+1]>=Ti,255,0)

            elif (d4[i[0]-1,i[1]-1]==1):
                In2[i[0]-1,i[1]-1]=np.where(In[i[0]-1,i[1]-1]>=Ti,255,0)
                In2[i[0]+1,i[1]+1]=np.where(In[i[0]+1,i[1]+1]>=Ti,255,0)
    toplevel=tkinter.Toplevel(window)
    toplevel.geometry("600x600")
    label=tkinter.Label(toplevel)
    label.pack(expand=True)
    img = Image.fromarray(In2)
    imgtk = ImageTk.PhotoImage(image=img)
    label.configure(image=imgtk)
    label.image=imgtk
    def func_save1():
        fileName=filedialog.askopenfilename( initialdir='/',title='파일선택',filetypes=(("","*.jpg"), ("all files", "*.*")) )
        cv2.imwrite(fileName,imageValue1)
    def func_exit1():
        toplevel.quit()
        toplevel.destroy()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="저장",command=func_save1)
    button.pack()
    button = tkinter.Button(toplevel, overrelief="solid", width=15,text="취소",command=func_exit1)
    button.pack()
label.pack(expand=True)
menubar=tkinter.Menu(window)
menu=tkinter.Menu(menubar, tearoff=0)
ImageDotMenu=tkinter.Menu(menubar, tearoff=0)
ImageSpaceMenu=tkinter.Menu(menubar, tearoff=0)
ImageDotMenu.add_command(label="히스토그램",command=func_calcHisto)
ImageDotMenu.add_command(label="히스토그램 균등화",command=func_equalHisto)
ImageSpaceMenu.add_command(label="가우시안필터",command=func_GaussF)
ImageSpaceMenu.add_command(label="1차미분필터",command=func_1stDF)
ImageSpaceMenu.add_command(label="DoG필터",command=func_dogF)
ImageSpaceMenu.add_command(label="캐니검출",command=func_canny)
menu.add_command(label="열기",command=func_open)
menu.add_command(label="저장하기",command=func_save)
menu.add_separator()
menu.add_command(label="끝내기", command=func_exit)
menubar.add_cascade(label="파일", menu=menu)
menubar.add_cascade(label="영상점처리", menu=ImageDotMenu)
menubar.add_cascade(label="영상공간처리", menu=ImageSpaceMenu)
window.config(menu=menubar)
window.mainloop()

