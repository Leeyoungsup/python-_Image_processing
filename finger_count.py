import numpy as np
from tkinter import *
from PIL import ImageTk,Image
import tkinter.filedialog as tkfile
from tkinter import messagebox
import math
import binarization as bin
def func_exit() :#창닫기
    window.quit()
    window.destroy()
def temp_open():#템플릿파일 열기
    global tfile
    filename=tkfile.askopenfilenames(title = "Select file",parent = window, filetypes = (("jpg files","*.jpg"),("all files","*.*")))
    tfile=np.array(filename)
    tfile1=np.copy(tfile)
    if tfile.shape[0]==10:
        temp_output()
    else:
        messagebox.showinfo("error", "사진을 10개 선택하여야합니다")
        temp_open()
    tfile=np.zeros((10,200,200))
    for i in range(10):
        tfile[i]=temp_q(tfile1,i)
    return filename
def X_open():
    global pfile,pphotoList
    filename=tkfile.askopenfilenames(title = "Select file",parent = window, filetypes = (("jpg files","*.jpg"),("all files","*.*")))
    pfile=np.array(filename)
    a=pfile.shape[0]
    pphotoList=[None]*a
    for i in range(a):
        pphotoList[i]=ImageTk.PhotoImage(Image.open(pfile[i]))
    templabel=Label(window,image=pphotoList[0])
    templabel.place(x =0, y = 300 )
    return filename
def temp_q(file,i):
    global k1
    k=np.array(Image.open(file[i]))
    k=cal.Lsm(k)
    k=cal.otsu(k)
    k=cal.practice(k)
    k=cal.otsu(k)
    k=cal.temp_pca(k)
    k=cal.s_delete(k)
    k=cal.rrr(k)
    return k
def finger_joint_count(file,i):
    imsi=np.array(Image.open(file[i]))
    imsi=cal.Lsm(imsi)
    imsi=cal.otsu(imsi)
    imsi4=np.copy(imsi)
    imsi4=cal.practice(imsi4)
    imsi2=cal.Morphology_Erosion(imsi)
    imsi=imsi-imsi2
    point=cal.convexHull(imsi)
    a=cal.finger_joint_detection(imsi4,point)
    return a
def temp_output():
    global num,xPos,yPos,tfile,window,photoList
    for i in range(10) :
        photoList[i] = ImageTk.PhotoImage(Image.open(tfile[i]).resize((100,100),Image.ANTIALIAS))
    for i in range(0, 2) : 
        for k in range(0, 5) :
            templabel=Label(window,image=photoList[num])
            templabel.place(x = xPos, y = yPos)
            num+=1
            xPos+=110
        xPos = 0
        yPos +=110
def template_matching():
    global pfile,rnum,valuememory,pphotoList
    sha=pfile.shape[0]
    rnum1=rnum%sha
    listbox2.delete (0, END)
    templabel=Label(window,image=pphotoList[rnum1])
    templabel.place(x =0, y = 300 )
    tempin=np.zeros((sha))
    for i in range(10):
        a=tfile[i]
        b=temp_q(pfile,rnum1)
        k=abs(b-a)
        value=k.sum()
        valuememory[i]=value
        if i==0:
            listbox1.delete (0, END)
            listbox1.insert (0, value )
        else:
            listbox1.insert (END, value )
    c=np.array(np.where(valuememory==np.min(valuememory)))
    tempin[rnum1]=c
    listbox2.insert (0,pfile[rnum1][-10:]+"->"+str(c[0]+1))
    rnum+=1
def template_matchingall():
    global pfile,rnum,valuememory,insik,pphotoList
    sha=pfile.shape[0]
    listbox2.delete (0, END)
    tempin=np.zeros((10))
    tw=0
    count=np.zeros((10))
    for j in range(sha):
        templabel=Label(window,image=pphotoList[j])
        templabel.place(x =0, y = 300 )
        for i in range(10):
            a=tfile[i]
            b=temp_q(pfile,j)
            k=abs(b-a)
            value=k.sum()
            valuememory[i]=value
            if i==0:
                listbox1.delete (0, END)
                listbox1.insert (0, value )
            else:
                listbox1.insert (END, value )
        c=np.array(np.where(valuememory==np.min(valuememory)))
        listbox2.insert (sha,pfile[j][-10:]+"->"+str(c[0]+1))
        for ch in range(1,len(pfile[j])):
            if pfile[j][-ch]=='_':
                tw=-ch-1
                break
        if int(pfile[j][tw])==0:
            count[int(pfile[j][tw])+9]+=1
            if int(pfile[j][tw])==c[0,0]:
                tempin[int(pfile[j][tw])+9]+=1
        else:
            count[int(pfile[j][tw])-1]+=1
            if int(pfile[j][tw])-1==c[0,0]:
                tempin[int(pfile[j][tw])-1]+=1
    for i in range(10):
        insik=tempin[i]/count[i]*100
        listbox2.insert (sha+i+1,'T'+str(i+1)+'인식률='+str(insik)+'%')
        if i==9:
            insik=tempin.sum()/count.sum()*100
            listbox2.insert (sha+i+2,'전체인식률='+str(insik)+'%')
def save_list():

    temp_list = list(listbox2.get(0,END))
    temp_list = [chem + '\n' for chem in temp_list]
    fout = open("하기싫다.txt", "w")
    fout.writelines(temp_list)
    fout.close()
def finger_count():
    global pfile,rnum,valuememory,pphotoList
    sha=pfile.shape[0]
    rnum1=rnum%sha
    point=finger_joint_count(pfile,rnum1)
    listbox2.delete (0, END)
    templabel=Label(window,image=pphotoList[rnum1])
    templabel.place(x =0, y = 300 )
    listbox2.insert (0,pfile[rnum1][-10:]+"->"+str(point))
    rnum+=1
def finger_count_all():
    global pfile,rnum,valuememory,insik,pphotoList,count
    sha=pfile.shape[0]
    listbox2.delete (0, END)
    tempin=np.zeros((10))
    tw=0
    count=np.zeros((10))
    for j in range(sha):
        point=finger_joint_count(pfile,j)
        templabel=Label(window,image=pphotoList[j])
        templabel.place(x =0, y = 300 )
        listbox2.insert (sha,pfile[j][-10:]+"->"+str(point))
        for ch in range(1,len(pfile[j])):
            if pfile[j][-ch]=='_':
                tw=-ch-1
                break
        if int(pfile[j][tw])==0:
            count[int(pfile[j][tw])+9]+=1
            if point==0:
                tempin[int(pfile[j][tw])+9]+=1
        elif int(pfile[j][tw])==1 or int(pfile[j][tw])==9:
            count[int(pfile[j][tw])-1]+=1
            if point==1:
                tempin[int(pfile[j][tw])-1]+=1
        elif int(pfile[j][tw])==2 or int(pfile[j][tw])==6 or int(pfile[j][tw])==8:
            count[int(pfile[j][tw])-1]+=1
            if point==2:
                tempin[int(pfile[j][tw])-1]+=1
        elif int(pfile[j][tw])==3:
            count[int(pfile[j][tw])-1]+=1
            if point==3:
                tempin[int(pfile[j][tw])-1]+=1
        elif int(pfile[j][tw])==4:
            count[int(pfile[j][tw])-1]+=1
            if point==4:
                tempin[int(pfile[j][tw])-1]+=1
        elif int(pfile[j][tw])==5:
            count[int(pfile[j][tw])-1]+=1
            if point==5:
                tempin[int(pfile[j][tw])-1]+=1
        elif int(pfile[j][tw])==7:
            count[int(pfile[j][tw])-1]+=1
            if point==0:
                tempin[int(pfile[j][tw])-1]+=1
    for i in range(10):
        insik=tempin[i]/count[i]*100
        listbox2.insert (sha+i+1,'T'+str(i+1)+'인식률='+str(insik)+'%')
        if i==9:
            insik=tempin.sum()/count.sum()*100
            listbox2.insert (sha+i+2,'전체인식률='+str(insik)+'%')
cal=bin.dinaization()
window = Tk()
frame=Frame(window)
frame1=Frame(window)
tfile,pfile=0,0
num,rnum=0,0
xPos, yPos = 0, 0
value,insik=0,0
valuememory=np.zeros((10))
photoList,pphotoList=[None] *10,0
btn=Button(window,text="TM_RUN!",command=template_matching)
btn.config(width=10, height=5)
btn1=Button(window,text="TM_RUN ALL!",command=template_matchingall)
btn1.config(width=10, height=5)
btn.place(x=220,y=230)
btn1.place(x=220,y=330)
btn2=Button(window,text="FC_RUN!",command=finger_count)
btn2.config(width=10, height=5)
btn3=Button(window,text="FC_RUN ALL!",command=finger_count_all)
btn3.config(width=10, height=5)
btn2.place(x=320,y=230)
btn3.place(x=320,y=330)
mainMenu = Menu(window)
window.config(menu = mainMenu)
fileMenu = Menu(mainMenu)
window.geometry("800x600")
mainMenu.add_cascade(label = "파일", menu = fileMenu)
fileMenu.add_command(label = "템플릿 등록하기",command=temp_open)
fileMenu.add_command(label = "인식파일 불러오기",command=X_open)
fileMenu.add_command(label = "실행결과 저장",command=save_list)
fileMenu.add_command(label = "끝내기",command=func_exit)
window.title("머신러닝1_LYS_1558021")
scrollbar=Scrollbar(frame)
scrollbar.pack(side="right", fill="y")
scrollbar1=Scrollbar(frame1)
scrollbar1.pack(side="right", fill="y")
listbox1= Listbox(frame,width=30, height=15,yscrollcommand=scrollbar.set)
listbox1.pack()
scrollbar["command"]=listbox1.yview
listbox2= Listbox(frame1,width=30, height=15,yscrollcommand=scrollbar1.set)
listbox2.pack()
scrollbar1["command"]=listbox2.yview
frame.pack()
frame.place(x=550,y=0)
frame1.pack()
frame1.place(x=550,y=300)
window.mainloop()
