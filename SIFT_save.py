<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter.filedialog as tkfile
from PIL import ImageTk, Image
from skimage import data, img_as_float

def Gaussian_filter(image, sigma):
    sn = 15
    sn1 = sn // 2
    x = np.arange(-sn1, sn1 + 1)
    y = np.arange(-sn1, sn1 + 1)
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    height = image.shape[0]
    width = image.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1 * 2, width + sn1 * 2])
    imageValue1 = np.zeros([height, width])
    imageValue[y + sn1, x + sn1] = image[y, x]
    for sn2 in range(sn1):
        imageValue[sn1:-sn1, sn2] = image[:, 0]
        imageValue[sn1:-sn1, -sn2 - 1] = image[:, -1]
    for sn2 in range(sn1):
        imageValue[sn2] = imageValue[sn1]
        imageValue[-sn2 - 1] = imageValue[-sn1 - 1]

    for i in range(sn1, height + sn1):
        for j in range(sn1, width + sn1):
            value = Mask * imageValue[i - sn1:i + sn1 + 1, j - sn1:j + sn1 + 1]
            imageValue1[i - sn1, j - sn1] = np.sum(value)
    return imageValue1


def Scale(imsi, Magnification):
    imsi1 = np.zeros([int(imsi.shape[0] * Magnification), int(imsi.shape[1] * Magnification)])
    Magnification = np.array([[Magnification, 0], [0, Magnification]])
    for x1 in range(imsi.shape[1]):
        for y1 in range(imsi.shape[0]):
            xy = np.array([[y1], [x1]])
            xy = np.dot(Magnification, xy)
            xy = xy.astype(np.int64)
            imsi1[xy[0], xy[1]] = imsi[y1, x1]
    return imsi1


def harrisCorner(imsi, xy):
    height = imsi.shape[0]
    width = imsi.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    sn = 3
    sn1 = (sn - 1) // 2
    imsi1 = np.zeros([height + sn - 1, width + sn - 1])
    imsi1[y + sn1, x + sn1] = imsi[y, x]
    for i in range(sn1):
        imsi1[:, i] = imsi1[:, sn1]
        imsi1[:, -(i + 1)] = imsi1[:, -(sn1 + 1)]
    for i in range(sn1):
        imsi1[i] = imsi1[sn1]
        imsi1[-(i + 1)] = imsi1[-(sn1 + 1)]
    a = np.zeros([imsi1.shape[0], 1])
    b = np.zeros([1, imsi1.shape[1]])
    Ix = imsi1[:, 1:] - imsi1[:, :-1]
    Ix = np.hstack((a, Ix))
    k = 0.04
    Iy = imsi1[1:] - imsi1[:-1]
    Iy = np.vstack((b, Iy))
    imsi2 = np.zeros_like(imsi)
    xy = xy.T
    for yx in xy:
        M = np.array([[np.sum((Ix[yx[0], yx[1] - sn1:yx[1] + sn1 + 1]) ** 2),
                       np.sum((Ix[yx[0], yx[1] - sn1:yx[1] + sn1 + 1] * Iy[yx[0] - sn1:yx[0] + sn1 + 1, yx[1]]))],
                      [np.sum((Ix[yx[0], yx[1] - sn1:yx[1] + sn1 + 1] * Iy[yx[0] - sn1:yx[0] + sn1 + 1, yx[1]])),
                       np.sum((Iy[yx[0] - sn1:yx[0] + sn1 + 1, yx[1]]) ** 2)]])
        R = np.linalg.det(M) - k * (np.trace(M) ** 2)
        imsi2[yx[0], yx[1]] = R
    imsi2_xy = np.where(imsi2 > (imsi2.max() * (15 / 100)))
    return np.array(imsi2_xy)


def median_filter(image):
    Mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    height = image.shape[0]
    width = image.shape[1]
    imageValue = np.zeros([height + 2, width + 2])
    imageValue1 = np.zeros([height + 2, width + 2])
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue[y + 1, x] = image[y, x]
    imageValue[y + 1, x + 2] = image[y, x]
    imageValue[0] = imageValue[1]
    imageValue[-1] = imageValue[-2]
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            value = Mask * imageValue[i - 1:i + 2, j - 1:j + 2]
            imageValue1[i, j] = np.sort(value.reshape(9))[8]
    imageValue1 = np.delete(imageValue1, width + 1, axis=1)
    imageValue1 = np.delete(imageValue1, height + 1, axis=0)
    imageValue1 = np.delete(imageValue1, 0, axis=1)
    imageValue1 = np.delete(imageValue1, 0, axis=0)
    return imageValue1


def DoG(oct1, oct2, oct3, oct4):
    for i in range(4):
        if i == 0:
            dog_oct1 = np.zeros([4, oct1.shape[1], oct1.shape[2]])
        if i == 1:
            dog_oct2 = np.zeros([4, oct2.shape[1], oct2.shape[2]])
        if i == 2:
            dog_oct3 = np.zeros([4, oct3.shape[1], oct3.shape[2]])
        if i == 3:
            dog_oct4 = np.zeros([4, oct4.shape[1], oct4.shape[2]])
        for j in range(4):
            if i == 0:
                dog_oct1[j] = oct1[j] - oct1[j + 1]
            if i == 1:
                dog_oct2[j] = oct2[j] - oct2[j + 1]
            if i == 2:
                dog_oct3[j] = oct3[j] - oct3[j + 1]
            if i == 3:
                dog_oct4[j] = oct4[j] - oct4[j + 1]
    return dog_oct1, dog_oct2, dog_oct3, dog_oct4


def keypoint_detection(dog_oct1, dog_oct2, dog_oct3, dog_oct4):
    for i in range(4):
        if i == 0:
            imsi2 = dog_oct1
            key_imsi1 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        elif i == 1:
            imsi2 = dog_oct2
            key_imsi2 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        elif i == 2:
            imsi2 = dog_oct3
            key_imsi3 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        elif i == 3:
            imsi2 = dog_oct4
            key_imsi4 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        for j in range(1, 3):
            if i == 0:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi1[j - 1][x, y] = 1
            elif i == 1:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi2[j - 1][x, y] = 1
            elif i == 2:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi3[j - 1][x, y] = 1
            elif i == 3:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi4[j - 1][x, y] = 1
    return key_imsi1, key_imsi2, key_imsi3, key_imsi4


def Gaussian_filter_Mask(sigma):
    sn = 15
    sn1 = sn // 2
    x = np.arange(-sn1, sn1 + 1)
    y = np.arange(-sn1, sn1 + 1)
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return Mask[sn1]


def OCT(imsi):
    k = np.sqrt(2)
    for oct in range(4):
        sigma = 1 / np.sqrt(2)
        if oct == 0:
            imsi1 = Scale(imsi, 2)
            imsi2 = median_filter(imsi1)
            oct1 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])
        elif oct == 1:
            imsi1 = Scale(imsi, 1)
            imsi2 = median_filter(imsi1)
            oct2 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])

        elif oct == 2:
            imsi1 = Scale(imsi, 0.5)
            imsi2 = median_filter(imsi1)
            oct3 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])

        elif oct == 3:
            imsi1 = Scale(imsi, 0.25)
            imsi2 = median_filter(imsi1)
            oct4 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])

        for i in range(5):
            if oct == 0:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct1[i] = imsi1
                sigma = k * sigma
            elif oct == 1:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct2[i] = imsi1
                sigma = k * sigma
            elif oct == 2:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct3[i] = imsi1
                sigma = k * sigma
            elif oct == 3:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct4[i] = imsi1
                sigma = k * sigma
    return oct1, oct2, oct3, oct4
def keypoint_Descriptor(keypoint, imsi):
    SIFT_histogram=np.zeros((128,1))
    height = imsi.shape[0]
    width = imsi.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    sn = 16
    sn1 = sn // 2
    imsi_ss = np.zeros([height + sn - 1, width + sn - 1])
    imsi_ss[y + sn1, x + sn1] = imsi[y, x]
    m = np.zeros((keypoint[0].shape[0], 16, 16))
    theta = np.zeros((keypoint[0].shape[0], 16, 16))
    keypoint = keypoint + sn1
    keypoint = keypoint.astype(np.int64)
    for i in range(sn1):
        imsi_ss[:, i] = imsi_ss[:, sn1]
        imsi_ss[:, -(i + 1)] = imsi_ss[:, -(sn1 + 1)]
    for i in range(sn1):
        imsi_ss[i] = imsi_ss[sn1]
        imsi_ss[-(i + 1)] = imsi_ss[-(sn1 + 1)]
    for i in range(keypoint[0].shape[0]):

        for window_x in range(16):
            for window_y in range(16):
                m[i, window_y, window_x] = np.sqrt((imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                    0, i] + window_x - 7 + 1] - imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                    0, i] + window_x - 7 - 1]) ** 2 + (imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                    0, i] + window_x - 7] - imsi_ss[keypoint[1, i] + window_y - 7 - 1, keypoint[
                    0, i] + window_x - 7]) ** 2)
                if (imsi_ss[keypoint[1, i] + window_y - 7, keypoint[0, i] + window_x - 7 + 1] - imsi_ss[
                    keypoint[1, i] + window_y - 7, keypoint[0, i] + window_x - 7 - 1]) == 0:
                    if (imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                                                                     0, i] + window_x - 7] - imsi_ss[
                            keypoint[1, i] + window_y - 7 - 1, keypoint[
                                                                   0, i] + window_x - 7]) > 0:
                        theta[i, window_y, window_x] = 90
                    else:

                        theta[i, window_y, window_x] = -90

                else:

                    theta[i, window_y, window_x] = np.degrees(
                        np.arctan((imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                            0, i] + window_x - 7] - imsi_ss[keypoint[1, i] + window_y - 7 - 1, keypoint[
                            0, i] + window_x - 7]) / (imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                            0, i] + window_x - 7 + 1] - imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                            0, i] + window_x - 7 - 1])))

    theta=theta+90
    for i in range(keypoint[0].shape[0]):
        SIFT_histogram1 = np.zeros((128, 1))
        for window_row in range(4):
            for window_colunm in range(4):
                window_theta=theta[i][window_row*4:(window_row+1)*4,window_colunm*4:(window_colunm+1)*4]
                window_m = m[i][window_row * 4:(window_row + 1) * 4, window_colunm * 4:(window_colunm + 1) * 4]
                for window_y in range(4):
                    for window_x in range(4):
                        SIFT_index=int(window_theta[window_y,window_x]//22.5)
                        if SIFT_index==8:
                            SIFT_index=7
                        SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]=SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]+window_m[window_y,window_x]
        SIFT_histogram=np.hstack((SIFT_histogram,SIFT_histogram1))
    SIFT_histogram=SIFT_histogram.T
    return SIFT_histogram

window = Tk()
filename=tkfile.askopenfilenames(title = "Select file",parent = window, filetypes = (("jpg files","*.jpg"),("all files","*.*")))
pfile=np.array(filename)
a=pfile.shape[0]
for file in range(a):
    imsi = np.array(Image.open(pfile[file])).astype(np.float32)
    oct1, oct2, oct3, oct4 = OCT(imsi)
    dog_oct1, dog_oct2, dog_oct3, dog_oct4 = DoG(oct1, oct2, oct3, oct4)
    key_imsi1, key_imsi2, key_imsi3, key_imsi4 = keypoint_detection(dog_oct1, dog_oct2, dog_oct3, dog_oct4)
    '''fig = plt.figure(figsize=(10, 10))
    fig.patch.set_visible(False)
    
    for i in range(4):
        for j in range(4):
            ax1 = fig.add_subplot(4, 4, i + 1 + j * 4)
            if j == 0:
    
                ax1.imshow(dog_oct1[i], 'gray')
            elif j == 1:
    
                ax1.imshow(dog_oct2[i], 'gray')
            elif j == 2:
    
                ax1.imshow(dog_oct3[i], 'gray')
            elif j == 3:
    
                ax1.imshow(dog_oct4[i], 'gray')
            plt.axis('off'), plt.xticks([]), plt.yticks([])'''
    all_point = np.array([[], []])
    for i in range(2):
        for j in range(4):
            #ax1 = fig.add_subplot(4, 4, i + 2 + j * 4)
            if j == 0:
                a = np.array(np.where(key_imsi1[i] == 1))
                b = harrisCorner(dog_oct1[i + 1], a)
                all_point = np.hstack((all_point, b // 2))
            elif j == 1:
                a = np.array(np.where(key_imsi2[i] == 1))
                b = harrisCorner(dog_oct2[i + 1], a)
                all_point = np.hstack((all_point, b))
            elif j == 2:
                a = np.array(np.where(key_imsi3[i] == 1))
                b = harrisCorner(dog_oct3[i + 1], a)
                all_point = np.hstack((all_point, b * 2))
            elif j == 3:
                a = np.array(np.where(key_imsi4[i] == 1))
                b = harrisCorner(dog_oct4[i + 1], a)
                all_point = np.hstack((all_point, b * 4))
            '''ax1.plot(b[1], b[0], 'r+')
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.figure()
    plt.plot(all_point[1], all_point[0], 'r+')
    plt.imshow(imsi, 'gray')
    plt.show()'''
    keypoint=np.copy(all_point)
    SIFT_histogram=np.zeros((128,1))
    height = imsi.shape[0]
    width = imsi.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    sn = 16
    sn1 = sn//2
    imsi_ss = np.zeros([height + sn , width + sn])
    imsi_ss[y + sn1, x + sn1] = imsi[y, x]
    m = np.zeros((keypoint[0].shape[0], 16, 16))
    theta = np.zeros((keypoint[0].shape[0], 16, 16))

    keypoint = keypoint + sn1
    keypoint = keypoint.astype(np.int64)
    for i in range(sn1):
        imsi_ss[:, i] = imsi_ss[:, sn1]
        imsi_ss[:, -(i + 1)] = imsi_ss[:, -(sn1 + 1)]
    for i in range(sn1):
        imsi_ss[i] = imsi_ss[sn1]
        imsi_ss[-(i + 1)] = imsi_ss[-(sn1 + 1)]
    for i in range(keypoint[0].shape[0]):

        for window_x in range(16):
            for window_y in range(16):
                m[i, window_y, window_x] = np.sqrt((imsi_ss[keypoint[1, i] + window_y - 8, keypoint[
                    0, i] + window_x - 8 + 1] - imsi_ss[keypoint[1, i] + window_y - 8, keypoint[
                    0, i] + window_x - 8 - 1]) ** 2 + (imsi_ss[keypoint[1, i] + window_y - 8 + 1, keypoint[
                    0, i] + window_x - 8] - imsi_ss[keypoint[1, i] + window_y - 8 - 1, keypoint[
                    0, i] + window_x - 8]) ** 2)
                if (imsi_ss[keypoint[1, i] + window_y - 8, keypoint[0, i] + window_x - 8 + 1] - imsi_ss[
                    keypoint[1, i] + window_y - 8, keypoint[0, i] + window_x - 8 - 1]) == 0:
                    if (imsi_ss[keypoint[1, i] + window_y - 8 + 1, keypoint[
                                                                       0, i] + window_x - 8] - imsi_ss[
                            keypoint[1, i] + window_y - 8 - 1, keypoint[
                                                                   0, i] + window_x - 8]) > 0:
                        theta[i, window_y, window_x] = 90
                    else:

                        theta[i, window_y, window_x] = -90

                else:

                    theta[i, window_y, window_x] = np.degrees(
                        np.arctan((imsi_ss[keypoint[1, i] + window_y - 8 + 1, keypoint[
                            0, i] + window_x - 8] - imsi_ss[keypoint[1, i] + window_y - 8 - 1, keypoint[
                            0, i] + window_x - 8]) / (imsi_ss[keypoint[1, i] + window_y - 8, keypoint[
                            0, i] + window_x - 8 + 1] - imsi_ss[keypoint[1, i] + window_y - 8, keypoint[
                            0, i] + window_x - 8 - 1])))

    theta=theta+90
    for i in range(keypoint[0].shape[0]):
        SIFT_histogram1 = np.zeros((128, 1))
        for window_row in range(4):
            for window_colunm in range(4):
                window_theta=theta[i][window_row*4:(window_row+1)*4,window_colunm*4:(window_colunm+1)*4]
                window_m = m[i][window_row * 4:(window_row + 1) * 4, window_colunm * 4:(window_colunm + 1) * 4]
                for window_y in range(4):
                    for window_x in range(4):
                        SIFT_index=int(window_theta[window_y,window_x]//22.5)
                        if SIFT_index==8:
                            SIFT_index=7
                        SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]=SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]+window_m[window_y,window_x]
        SIFT_histogram=np.hstack((SIFT_histogram,SIFT_histogram1))
    SIFT_histogram=SIFT_histogram.T
    SIFT_histogram=np.delete(SIFT_histogram,0,axis=0)
    np.save("./SIFT_max(15 100)_save/4/"+str(file)+".npy",SIFT_histogram)
    print(file)
=======
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image
from skimage import data, img_as_float


def Gaussian_filter(image, sigma):
    sn = 15
    sn1 = sn // 2
    x = np.arange(-sn1, sn1 + 1)
    y = np.arange(-sn1, sn1 + 1)
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    height = image.shape[0]
    width = image.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue = np.zeros([height + sn1 * 2, width + sn1 * 2])
    imageValue1 = np.zeros([height, width])
    imageValue[y + sn1, x + sn1] = image[y, x]
    for sn2 in range(sn1):
        imageValue[sn1:-sn1, sn2] = image[:, 0]
        imageValue[sn1:-sn1, -sn2 - 1] = image[:, -1]
    for sn2 in range(sn1):
        imageValue[sn2] = imageValue[sn1]
        imageValue[-sn2 - 1] = imageValue[-sn1 - 1]

    for i in range(sn1, height + sn1):
        for j in range(sn1, width + sn1):
            value = Mask * imageValue[i - sn1:i + sn1 + 1, j - sn1:j + sn1 + 1]
            imageValue1[i - sn1, j - sn1] = np.sum(value)
    return imageValue1


def Scale(imsi, Magnification):
    imsi1 = np.zeros([int(imsi.shape[0] * Magnification), int(imsi.shape[1] * Magnification)])
    Magnification = np.array([[Magnification, 0], [0, Magnification]])
    for x1 in range(imsi.shape[1]):
        for y1 in range(imsi.shape[0]):
            xy = np.array([[y1], [x1]])
            xy = np.dot(Magnification, xy)
            xy = xy.astype(np.int64)
            imsi1[xy[0], xy[1]] = imsi[y1, x1]
    return imsi1


def harrisCorner(imsi, xy):
    height = imsi.shape[0]
    width = imsi.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    sn = 3
    sn1 = (sn - 1) // 2
    imsi1 = np.zeros([height + sn - 1, width + sn - 1])
    imsi1[y + sn1, x + sn1] = imsi[y, x]
    for i in range(sn1):
        imsi1[:, i] = imsi1[:, sn1]
        imsi1[:, -(i + 1)] = imsi1[:, -(sn1 + 1)]
    for i in range(sn1):
        imsi1[i] = imsi1[sn1]
        imsi1[-(i + 1)] = imsi1[-(sn1 + 1)]
    a = np.zeros([imsi1.shape[0], 1])
    b = np.zeros([1, imsi1.shape[1]])
    Ix = imsi1[:, 1:] - imsi1[:, :-1]
    Ix = np.hstack((a, Ix))
    k = 0.04
    Iy = imsi1[1:] - imsi1[:-1]
    Iy = np.vstack((b, Iy))
    imsi2 = np.zeros_like(imsi)
    xy = xy.T
    for yx in xy:
        M = np.array([[np.sum((Ix[yx[0], yx[1] - sn1:yx[1] + sn1 + 1]) ** 2),
                       np.sum((Ix[yx[0], yx[1] - sn1:yx[1] + sn1 + 1] * Iy[yx[0] - sn1:yx[0] + sn1 + 1, yx[1]]))],
                      [np.sum((Ix[yx[0], yx[1] - sn1:yx[1] + sn1 + 1] * Iy[yx[0] - sn1:yx[0] + sn1 + 1, yx[1]])),
                       np.sum((Iy[yx[0] - sn1:yx[0] + sn1 + 1, yx[1]]) ** 2)]])
        R = np.linalg.det(M) - k * (np.trace(M) ** 2)
        imsi2[yx[0], yx[1]] = R
    imsi2_xy = np.where(imsi2 > (imsi2.max() * (15 / 100)))
    return np.array(imsi2_xy)


def median_filter(image):
    Mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    height = image.shape[0]
    width = image.shape[1]
    imageValue = np.zeros([height + 2, width + 2])
    imageValue1 = np.zeros([height + 2, width + 2])
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    imageValue[y + 1, x] = image[y, x]
    imageValue[y + 1, x + 2] = image[y, x]
    imageValue[0] = imageValue[1]
    imageValue[-1] = imageValue[-2]
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            value = Mask * imageValue[i - 1:i + 2, j - 1:j + 2]
            imageValue1[i, j] = np.sort(value.reshape(9))[8]
    imageValue1 = np.delete(imageValue1, width + 1, axis=1)
    imageValue1 = np.delete(imageValue1, height + 1, axis=0)
    imageValue1 = np.delete(imageValue1, 0, axis=1)
    imageValue1 = np.delete(imageValue1, 0, axis=0)
    return imageValue1


def DoG(oct1, oct2, oct3, oct4):
    for i in range(4):
        if i == 0:
            dog_oct1 = np.zeros([4, oct1.shape[1], oct1.shape[2]])
        if i == 1:
            dog_oct2 = np.zeros([4, oct2.shape[1], oct2.shape[2]])
        if i == 2:
            dog_oct3 = np.zeros([4, oct3.shape[1], oct3.shape[2]])
        if i == 3:
            dog_oct4 = np.zeros([4, oct4.shape[1], oct4.shape[2]])
        for j in range(4):
            if i == 0:
                dog_oct1[j] = oct1[j] - oct1[j + 1]
            if i == 1:
                dog_oct2[j] = oct2[j] - oct2[j + 1]
            if i == 2:
                dog_oct3[j] = oct3[j] - oct3[j + 1]
            if i == 3:
                dog_oct4[j] = oct4[j] - oct4[j + 1]
    return dog_oct1, dog_oct2, dog_oct3, dog_oct4


def keypoint_detection(dog_oct1, dog_oct2, dog_oct3, dog_oct4):
    for i in range(4):
        if i == 0:
            imsi2 = dog_oct1
            key_imsi1 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        elif i == 1:
            imsi2 = dog_oct2
            key_imsi2 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        elif i == 2:
            imsi2 = dog_oct3
            key_imsi3 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        elif i == 3:
            imsi2 = dog_oct4
            key_imsi4 = np.zeros([2, imsi2.shape[1], imsi2.shape[2]])
        for j in range(1, 3):
            if i == 0:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi1[j - 1][x, y] = 1
            elif i == 1:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi2[j - 1][x, y] = 1
            elif i == 2:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi3[j - 1][x, y] = 1
            elif i == 3:
                for x in range(1, imsi2[0].shape[1] - 1):
                    for y in range(1, imsi2[0].shape[0] - 1):
                        min = np.min(np.array(
                            [np.min(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.min(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.min(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        max = np.max(np.array(
                            [np.max(imsi2[j - 1][x - 1:x + 2, y - 1:y + 2]), np.max(imsi2[j][x - 1:x + 2, y - 1:y + 2]),
                             np.max(imsi2[j + 1][x - 1:x + 2, y - 1:y + 2])]))
                        if imsi2[j][x, y] == min or imsi2[j][x, y] == max:
                            key_imsi4[j - 1][x, y] = 1
    return key_imsi1, key_imsi2, key_imsi3, key_imsi4


def Gaussian_filter_Mask(sigma):
    sn = 15
    sn1 = sn // 2
    x = np.arange(-sn1, sn1 + 1)
    y = np.arange(-sn1, sn1 + 1)
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return Mask[sn1]


def OCT(imsi):
    k = np.sqrt(2)
    for oct in range(4):
        sigma = 1 / np.sqrt(2)
        if oct == 0:
            imsi1 = Scale(imsi, 2)
            imsi2 = median_filter(imsi1)
            oct1 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])
        elif oct == 1:
            imsi1 = Scale(imsi, 1)
            imsi2 = median_filter(imsi1)
            oct2 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])

        elif oct == 2:
            imsi1 = Scale(imsi, 0.5)
            imsi2 = median_filter(imsi1)
            oct3 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])

        elif oct == 3:
            imsi1 = Scale(imsi, 0.25)
            imsi2 = median_filter(imsi1)
            oct4 = np.zeros([5, imsi2.shape[1], imsi2.shape[0]])

        for i in range(5):
            if oct == 0:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct1[i] = imsi1
                sigma = k * sigma
            elif oct == 1:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct2[i] = imsi1
                sigma = k * sigma
            elif oct == 2:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct3[i] = imsi1
                sigma = k * sigma
            elif oct == 3:
                imsi1 = Gaussian_filter(imsi2, sigma)
                oct4[i] = imsi1
                sigma = k * sigma
    return oct1, oct2, oct3, oct4
def keypoint_Descriptor(keypoint, imsi):
    SIFT_histogram=np.zeros((128,1))
    height = imsi.shape[0]
    width = imsi.shape[1]
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    sn = 16
    sn1 = sn // 2
    imsi_ss = np.zeros([height + sn - 1, width + sn - 1])
    imsi_ss[y + sn1, x + sn1] = imsi[y, x]
    m = np.zeros((keypoint[0].shape[0], 16, 16))
    theta = np.zeros((keypoint[0].shape[0], 16, 16))
    keypoint = keypoint + sn1
    keypoint = keypoint.astype(np.int64)
    for i in range(sn1):
        imsi_ss[:, i] = imsi_ss[:, sn1]
        imsi_ss[:, -(i + 1)] = imsi_ss[:, -(sn1 + 1)]
    for i in range(sn1):
        imsi_ss[i] = imsi_ss[sn1]
        imsi_ss[-(i + 1)] = imsi_ss[-(sn1 + 1)]
    for i in range(keypoint[0].shape[0]):

        for window_x in range(16):
            for window_y in range(16):
                m[i, window_y, window_x] = np.sqrt((imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                    0, i] + window_x - 7 + 1] - imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                    0, i] + window_x - 7 - 1]) ** 2 + (imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                    0, i] + window_x - 7] - imsi_ss[keypoint[1, i] + window_y - 7 - 1, keypoint[
                    0, i] + window_x - 7]) ** 2)
                if (imsi_ss[keypoint[1, i] + window_y - 7, keypoint[0, i] + window_x - 7 + 1] - imsi_ss[
                    keypoint[1, i] + window_y - 7, keypoint[0, i] + window_x - 7 - 1]) == 0:
                    if (imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                                                                     0, i] + window_x - 7] - imsi_ss[
                            keypoint[1, i] + window_y - 7 - 1, keypoint[
                                                                   0, i] + window_x - 7]) > 0:
                        theta[i, window_y, window_x] = 90
                    else:

                        theta[i, window_y, window_x] = -90

                else:

                    theta[i, window_y, window_x] = np.degrees(
                        np.arctan((imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                            0, i] + window_x - 7] - imsi_ss[keypoint[1, i] + window_y - 7 - 1, keypoint[
                            0, i] + window_x - 7]) / (imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                            0, i] + window_x - 7 + 1] - imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                            0, i] + window_x - 7 - 1])))

    theta=theta+90
    for i in range(keypoint[0].shape[0]):
        SIFT_histogram1 = np.zeros((128, 1))
        for window_row in range(4):
            for window_colunm in range(4):
                window_theta=theta[i][window_row*4:(window_row+1)*4,window_colunm*4:(window_colunm+1)*4]
                window_m = m[i][window_row * 4:(window_row + 1) * 4, window_colunm * 4:(window_colunm + 1) * 4]
                for window_y in range(4):
                    for window_x in range(4):
                        SIFT_index=int(window_theta[window_y,window_x]//22.5)
                        if SIFT_index==8:
                            SIFT_index=7
                        SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]=SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]+window_m[window_y,window_x]
        SIFT_histogram=np.hstack((SIFT_histogram,SIFT_histogram1))
    SIFT_histogram=SIFT_histogram.T
    return SIFT_histogram


imsi = np.array(Image.open("1_5_1.jpg")).astype(np.float32)
oct1, oct2, oct3, oct4 = OCT(imsi)
dog_oct1, dog_oct2, dog_oct3, dog_oct4 = DoG(oct1, oct2, oct3, oct4)
key_imsi1, key_imsi2, key_imsi3, key_imsi4 = keypoint_detection(dog_oct1, dog_oct2, dog_oct3, dog_oct4)
fig = plt.figure(figsize=(10, 10))
fig.patch.set_visible(False)
all_point = np.array([[], []])
for i in range(4):
    for j in range(4):
        ax1 = fig.add_subplot(4, 4, i + 1 + j * 4)
        if j == 0:

            ax1.imshow(dog_oct1[i], 'gray')
        elif j == 1:

            ax1.imshow(dog_oct2[i], 'gray')
        elif j == 2:

            ax1.imshow(dog_oct3[i], 'gray')
        elif j == 3:

            ax1.imshow(dog_oct4[i], 'gray')
        plt.axis('off'), plt.xticks([]), plt.yticks([])

for i in range(2):
    for j in range(4):
        ax1 = fig.add_subplot(4, 4, i + 2 + j * 4)
        if j == 0:
            a = np.array(np.where(key_imsi1[i] == 1))
            b = harrisCorner(dog_oct1[i + 1], a)
            all_point = np.hstack((all_point, b // 2))
        elif j == 1:
            a = np.array(np.where(key_imsi2[i] == 1))
            b = harrisCorner(dog_oct2[i + 1], a)
            all_point = np.hstack((all_point, b))
        elif j == 2:
            a = np.array(np.where(key_imsi3[i] == 1))
            b = harrisCorner(dog_oct3[i + 1], a)
            all_point = np.hstack((all_point, b * 2))
        elif j == 3:
            a = np.array(np.where(key_imsi4[i] == 1))
            b = harrisCorner(dog_oct4[i + 1], a)
            all_point = np.hstack((all_point, b * 4))
        ax1.plot(b[1], b[0], 'r+')
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
plt.figure()
plt.plot(all_point[1], all_point[0], 'r+')
plt.imshow(imsi, 'gray')
plt.show()
keypoint=np.copy(all_point)
SIFT_histogram=np.zeros((128,1))
height = imsi.shape[0]
width = imsi.shape[1]
x = np.arange(width)
y = np.arange(height)
x, y = np.meshgrid(x, y)
sn = 16
sn1 = sn // 2
imsi_ss = np.zeros([height + sn - 1, width + sn - 1])
imsi_ss[y + sn1, x + sn1] = imsi[y, x]
m = np.zeros((keypoint[0].shape[0], 16, 16))
theta = np.zeros((keypoint[0].shape[0], 16, 16))

keypoint = keypoint + sn1
keypoint = keypoint.astype(np.int64)
for i in range(sn1):
    imsi_ss[:, i] = imsi_ss[:, sn1]
    imsi_ss[:, -(i + 1)] = imsi_ss[:, -(sn1 + 1)]
for i in range(sn1):
    imsi_ss[i] = imsi_ss[sn1]
    imsi_ss[-(i + 1)] = imsi_ss[-(sn1 + 1)]
for i in range(keypoint[0].shape[0]):

    for window_x in range(16):
        for window_y in range(16):
            m[i, window_y, window_x] = np.sqrt((imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                0, i] + window_x - 7 + 1] - imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                0, i] + window_x - 7 - 1]) ** 2 + (imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                0, i] + window_x - 7] - imsi_ss[keypoint[1, i] + window_y - 7 - 1, keypoint[
                0, i] + window_x - 7]) ** 2)
            if (imsi_ss[keypoint[1, i] + window_y - 7, keypoint[0, i] + window_x - 7 + 1] - imsi_ss[
                keypoint[1, i] + window_y - 7, keypoint[0, i] + window_x - 7 - 1]) == 0:
                if (imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                                                                   0, i] + window_x - 7] - imsi_ss[
                        keypoint[1, i] + window_y - 7 - 1, keypoint[
                                                               0, i] + window_x - 7]) > 0:
                    theta[i, window_y, window_x] = 90
                else:

                    theta[i, window_y, window_x] = -90

            else:

                theta[i, window_y, window_x] = np.degrees(
                    np.arctan((imsi_ss[keypoint[1, i] + window_y - 7 + 1, keypoint[
                        0, i] + window_x - 7] - imsi_ss[keypoint[1, i] + window_y - 7 - 1, keypoint[
                        0, i] + window_x - 7]) / (imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                        0, i] + window_x - 7 + 1] - imsi_ss[keypoint[1, i] + window_y - 7, keypoint[
                        0, i] + window_x - 7 - 1])))

theta=theta+90
for i in range(keypoint[0].shape[0]):
    SIFT_histogram1 = np.zeros((128, 1))
    for window_row in range(4):
        for window_colunm in range(4):
            window_theta=theta[i][window_row*4:(window_row+1)*4,window_colunm*4:(window_colunm+1)*4]
            window_m = m[i][window_row * 4:(window_row + 1) * 4, window_colunm * 4:(window_colunm + 1) * 4]
            for window_y in range(4):
                for window_x in range(4):
                    SIFT_index=int(window_theta[window_y,window_x]//22.5)
                    if SIFT_index==8:
                        SIFT_index=7
                    SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]=SIFT_histogram1[((window_row*4)+window_colunm)*8+SIFT_index]+window_m[window_y,window_x]
    SIFT_histogram=np.hstack((SIFT_histogram,SIFT_histogram1))
SIFT_histogram=SIFT_histogram.T
SIFT_histogram=np.delete(SIFT_histogram,0,axis=0)
>>>>>>> e964f5204f2ec843123dbe4e04871463c520597e
