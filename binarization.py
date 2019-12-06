import math
from mpl_toolkits import mplot3d  # mplot3d임포트
import PIL.Image as pilimg  # PIL.Image임포트
import numpy as np  # numpy임포트
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.fftpack import fft2,ifft2,dct,fft


class dinaization:
    def __init__(self):
        self.image = 0
        self.bins = 0
        self.his = 0

    def Lsm(self, image):  # 배경제거
        gray1 = np.copy(image)
        bx = np.zeros((4, 20, 20))
        by = np.zeros((4, 20, 20))
        x = np.array(image[0].shape[0])
        y = np.array(image[:, 0].shape[0])
        py = np.arange(y)
        px = np.arange(x)
        aa = np.zeros((x * y, 3))
        px, py = np.meshgrid(px, py)
        bx[0] = px[:20, :20]
        bx[1] = px[-20:, :20]
        bx[2] = px[:20, -20:]
        bx[3] = px[-20:, -20:]
        by[0] = py[:20, :20]
        by[1] = py[-20:, :20]
        by[2] = py[:20, -20:]
        by[3] = py[-20:, -20:]
        bbx = np.concatenate((bx[0], bx[1]), axis=0)
        bbx1 = np.concatenate((bx[2], bx[3]), axis=0)
        bbx = np.concatenate((bbx1, bbx), axis=1)
        bby = np.concatenate((by[0], by[1]), axis=0)
        bby1 = np.concatenate((by[2], by[3]), axis=0)
        bby = np.concatenate((bby1, bby), axis=1)
        aa = np.array((bbx.flatten(), bby.flatten(), np.ones(bbx[0].shape[0] * bby[0].shape[0]))).T
        aa = aa.astype(int)
        bb = np.array((image[aa[:, 0], aa[:, 1]]))
        paa = np.linalg.pinv(aa)
        self.r = paa.dot(bb)
        x = np.array(gray1[0].shape[0])
        y = np.array(gray1[:, 0].shape[0])
        py = np.arange(y)
        px = np.arange(x)
        aa = np.zeros((x * y, 3))
        px, py = np.meshgrid(px, py)
        self.aa = np.array((px.flatten(), py.flatten(), np.ones(x * y))).T
        bb = self.aa.dot(self.r)
        self.bb = bb.reshape(int(math.sqrt(np.shape(bb)[0])), int(math.sqrt(np.shape(bb)[0]))).T
        gray = gray1 - self.bb
        gray = gray - np.min(gray)
        return gray

    def otsu(self, image):
        self.image = image
        self.his, self.bins = np.histogram(image, bins=np.arange(257))
        it = np.arange(0, 256)
        result = np.zeros((256))
        for i in range(0, 256):
            a = np.sum(self.his[0:i]) / 40000
            b = np.sum(self.his[i + 1:256]) / 40000
            q = np.sum(self.his[0:i])
            w = np.sum(self.his[i + 1:256])
            if q == 0:
                u1 = 0
            else:
                u1 = np.sum(self.his[0:i + 1].dot(it[0:i + 1])) / q
            if w == 0:
                u2 = 0
            else:
                u2 = np.sum(self.his[i + 1:256].dot(it[i + 1:256])) / w
            result[i] = (a * b) * (u1 - u2) ** 2
        self.result = np.array(result)
        self.tvalue = np.argmax(result)
        tvalue = np.where(image > self.tvalue, 0, 255)
        return tvalue

    def Lsm_ys(self, gray):
        global a
        gray1 = np.copy(gray)
        a = np.zeros((4))
        a[0] = gray[:5, :5].mean()
        a[1] = gray[:5, -5:].mean()
        a[2] = gray[- 5:, :5].mean()
        a[3] = gray[- 5:, - 5:].mean()
        q = np.linspace(a[0], a[2], 200)
        w = np.linspace(a[1], a[3], 200)
        for i in range(200):
            gray[i] = np.linspace(q[i], w[i], 200)
        gray = gray.astype(int)
        return gray

    def practice(self, image):  # 연습
        for i in range(200):
            for j in range(200):
                if image[i, j] == 0:
                    image[i, j] = 0
                else:
                    image[i, j] = self.image[i, j]
        return image

    def temp_pca(self, x):
        size = np.shape(x[0])[0]
        pca = np.zeros((size + 100, size + 100))
        q = np.array(np.where(x != 0))
        m = np.mean(q, 1)
        m = m[:, np.newaxis]
        z = q - m
        nn = z.shape[1]
        C = z.dot(z.T) / nn
        L, v = np.linalg.eig(C)
        c = np.argsort(L)
        v = np.vstack((v[:, c[-1]], v[:, c[-2]]))
        y = v.dot(q)
        y = np.round(y)
        if np.min(y[0]) < 0:
            y[0] = y[0] - np.min(y[0])
        if np.min(y[1]) < 0:
            y[1] = y[1] - np.min(y[1])
        y = y.astype(int)
        pca[y[0], y[1]] = x[q[0], q[1]]
        pca1 = np.array(np.where(pca == 0))
        k1 = pca1[0].shape[0]
        for i in range(k1):
            if pca1[0, i] != 0 and pca1[1, i] != 0 and pca1[1, i] != 299 and pca1[0, i] != 299:
                if pca[pca1[0, i] + 1, pca1[1, i]] != 0 and pca[pca1[0, i], pca1[1, i] + 1] != 0 and pca[
                    pca1[0, i] - 1, pca1[1, i]] != 0 and pca[pca1[0, i], pca1[1, i] - 1] != 0:
                    pca[pca1[0, i], pca1[1, i]] = pca[pca1[0, i] + 1, pca1[1, i]]
        return pca

    def rrr(self, gray):
        a = np.where(gray == np.max(gray))
        b = np.zeros((4))
        b[0] = np.max(a[0])
        b[1] = np.max(a[1])
        b[2] = np.min(a[0])
        b[3] = np.min(a[1])
        b = b.astype(int)
        gray1 = np.zeros((200, 200))
        gray = gray[b[2]:b[0], b[3]:b[1]]
        px = np.arange(b[0] - b[2])
        py = np.arange(b[1] - b[3])
        px, py = np.meshgrid(px, py)
        q = 199 / np.max(px)
        w = 199 / np.max(py)
        qw = np.array(([q, 0], [0, w]))
        pxy = np.array((px.flatten(), py.flatten()))
        pxy1 = qw.dot(pxy)
        pxy1 = pxy1.astype(int)
        gray1[pxy1[0], pxy1[1]] = gray[pxy[0], pxy[1]]
        k = np.array(np.where(gray1 == 0))
        qw = np.linalg.inv(qw)
        k1 = qw.dot(k)
        k1 = k1.astype(int)
        gray1[k[0], k[1]] = gray[k1[0], k1[1]]
        k = np.array(np.where(gray1 == 0))
        gray1 = gray1
        k1 = k[0].shape[0]
        for i in range(k1):
            if k[0, i] != 0 and k[1, i] != 0 and k[1, i] != 199 and k[0, i] != 199:
                if gray1[k[0, i] + 1, k[1, i]] == 255 and gray1[k[0, i] - 1, k[1, i]] == 255:
                    gray1[k[0, i], k[1, i]] = 255
        return gray1

    def qqq(self, gray):
        t1 = gray[:, 0].shape[0]
        t2 = gray[0].shape[0]
        a = np.where(gray == np.max(gray))
        b = np.zeros((4))
        qq = np.where(gray != 0)[0].shape[0]
        t = t1 * t2
        qq = qq / t
        b[0] = np.max(a[0])
        b[1] = np.max(a[1])
        b[2] = np.min(a[0])
        b[3] = np.min(a[1])
        b = b.astype(int)
        gray1 = np.zeros((200, 200))
        gray = gray[b[2]:b[0], b[3]:b[1]]
        px = np.arange(b[0] - b[2])
        py = np.arange(b[1] - b[3])
        px, py = np.meshgrid(px, py)
        q = (199 / np.max(px))
        w = (199 / np.max(py))
        px1 = q * px * qq
        py1 = w * py * qq
        px1 = px1.astype(int)
        py1 = py1.astype(int)
        qy = 200 - np.max(py1)
        qx = 200 - np.max(px1)
        gray1[px1 + qx - 1, py1 + qy // 2] = gray[px, py]
        return gray1

    def s_delete(self, gray):
        a = []
        for i in range(0, 200):
            count = 0
            for j in range(0, 200):
                if gray[i, j] != 0:
                    count += 1
            a.append(count)
        a = np.array(a)
        am = np.argmax(a)
        S = int(a[am])
        c = int(am + S)
        gray[c:, :] = 0
        return gray

    def lys(self, gray):
        a = np.zeros([200])
        x = np.zeros([200])
        for i in range(0, 200):
            count = 0
            for j in range(0, 200):
                if gray[i, j] == 255:
                    count += 1
                    a[i] = count
                    if count == 1:
                        x[i] = j
                else:
                    count = 0
        am = np.argmax(a)
        S = int(a[am])
        gray1 = np.zeros([200, 200])
        self.amx = int(x[am] + S / 2)
        px = np.arange(200)
        py = np.arange(200)
        [px, py] = np.meshgrid(px, py)
        px = px - self.amx
        py = py - S
        px = px ** 2
        py = py ** 2
        self.pxy = px + py
        self.p_point = np.where(np.sqrt(self.pxy).astype(np.int64) == int(am / 2))
        self.gray1 = gray[self.p_point[0], self.p_point[1]]
        gray[self.p_point[0], self.p_point[1]] = 0
        return gray

    def d3_plot(self, imsi6):
        plt.figure()  # figure창 띄움
        ax = plt.axes(projection='3d')  # ax에 3d프로젝트 저장
        pmax = imsi6.max(axis=0).max(axis=0)  # imsi6의 가장큰값을 pmax에 저장
        pamax = np.where(imsi6 == pmax)  # imsi6의 pmax위치값을 pamax에 저장
        pmin = imsi6.min(axis=0).min(axis=0)  # imsi6의 가장 작은값을 pmin에 저장
        pamin = np.where(imsi6 == pmin)  # imsi6의 pmin위치값을 pamax에 저장
        x = np.arange(200)  # x에 0부터 480까지값 저장
        y = np.arange(200)  # y에 0부터 640까지의 값 저장
        [x, y] = np.meshgrid(x, y)  # x,y의 행렬크기를 같게함
        ax.scatter(pamax[1], pamax[0], pmax, c='r', marker='o')  # 좌표에 빨간색의 마커를 찍음
        ax.scatter(pamin[1], pamin[0], pmin, c='r', marker='o')  # 좌표에 빨간색의 마커를 찍음
        ax.plot_surface(x, y, imsi6, rstride=1, cstride=1, cmap='viridis', edgecolor='none')  # 3d 출력
        plt.show()  # 화면 출력

    def histogram(self, image):
        self.his, self.bins = np.histogram(image, bins=np.arange(257))
        self.it = np.arange(0, 256)
        self.k = image.astype(np.int64)
        for i in range(0, 256):
            p = np.sum(self.his[0:i]) / 40000
            s = p * 255
            self.k = np.where(self.k == i, s, self.k)
        return self.k

    def filter(self, image):
        Mask = np.array([[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]])
        height = image.shape[0]
        width = image.shape[1]
        imageValue = np.zeros([height + 2, width + 2])
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
                imageValue[i, j] = np.sort(value.reshape(9))[4]
        imageValue = np.delete(imageValue, width + 1, axis=1)
        imageValue = np.delete(imageValue, height + 1, axis=0)
        imageValue = np.delete(imageValue, 0, axis=1)
        imageValue = np.delete(imageValue, 0, axis=0)
        return imageValue

    def Morphology_Erosion(self, image):
        Mask = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]])
        height = image.shape[0]
        width = image.shape[1]
        imageValue = np.zeros([height, width])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if np.sum(image[i-1:i+2,j-1:j+2]*Mask)==255*5:
                    value = 255
                else:
                    value = 0
                imageValue[i, j] = value
        return imageValue

    def Morphology_Dilation(self, image):
        Mask = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]])
        height = image.shape[0]
        width = image.shape[1]
        imageValue = np.zeros([height, width])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if np.sum(image[i-1:i+2,j-1:j+2]*Mask)>=255:
                    value = 255
                else:
                    value = 0
                imageValue[i, j] = value

        return imageValue

    def Gray_Morphology_Erosion(self, image):
        Mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        height = image.shape[0]
        width = image.shape[1]
        imageValue = np.zeros([height, width])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                value = np.min(image[i - 1:i + 2, j - 1:j + 2] - Mask)
                imageValue[i, j] = value
        imageValue[0] = image[0]
        imageValue[height - 1] = image[height - 1]
        imageValue[:, width - 1] = image[:, width - 1]
        imageValue[:, 0] = image[:, 0]
        return imageValue

    def Gray_Morphology_Dilation(self, image):
        Mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        height = image.shape[0]
        width = image.shape[1]
        imageValue = np.zeros([height, width])
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                value = np.max(image[i - 1:i + 2, j - 1:j + 2] + Mask)
                imageValue[i, j] = value
        imageValue[0] = image[0]
        imageValue[height - 1] = image[height - 1]
        imageValue[:, width - 1] = image[:, width - 1]
        imageValue[:, 0] = image[:, 0]

        return imageValue

    def convexHull(self, image):
        points = np.array([np.where(image == 255)[1], np.where(image == 255)[0]])
        points = points.T
        hull = ConvexHull(points)
        Hull_Points = np.array([points[hull.vertices, 0], points[hull.vertices, 1]])
        return Hull_Points

    def finger_joint_detection(self, image, point):
        point1 = np.copy(point)
        center_point = np.array(
            [[int(np.mean(point1[0, np.where(point1[1] == np.max(point1[1]))[0]])), np.max(point1[1])]])
        center_point = center_point.T
        point = np.delete(point, np.where(point[1] == np.max(point[1]))[0], axis=1)
        a = point - center_point
        zero_a = np.where(a[0] == 0)[0]
        a = np.delete(a, zero_a, axis=1)
        point = np.delete(point, zero_a, axis=1)
        inclination = a[1] / a[0]
        d_point = np.array([])
        Angle = np.arctan(inclination) * 180 / np.pi + 180
        for i in range(Angle.shape[0]):
            for j in range(i + 1, Angle.shape[0]):
                if abs(Angle[i] - Angle[j]) <= 5:
                    d_point = np.append(d_point, j)
        inclination = np.delete(inclination, d_point)
        point = np.delete(point, d_point, axis=1)
        b = -inclination * center_point[0] + center_point[1]
        for i in range(b.shape[0]):
            y = np.arange(point[1][i], center_point[1][0])
            y = np.sort(y)[::-1]
            y = y.astype(np.int64)
            x = (-b[i] + y) / inclination[i]
            x = x.astype(np.int64)
            k = image[y, x]
            d_point = np.array([])
            d_k = k[1:] - k[:-1]
            count = 0
            p = 1
            while (p < d_k.shape[0] - 1):
                d_y1 = 0
                d_y2 = 0
                for q1 in np.sort(range(0, p)[::-1]):
                    if d_k[q1] <= 0:
                        d_y1 = d_y1 + d_k[q1]
                        pass
                    else:
                        break
                for q2 in range(p + 1, d_k.shape[0]):
                    if d_k[q2] >= 0:
                        d_y2 = d_y2 + d_k[q2]
                        pass
                    else:
                        break

                if d_y2 != 0 and d_y1 != 0 and abs(d_y1) + abs(d_y2) >= 15:
                    count += 1
                p = q2

            if count < 3:
                d_point = np.append(d_point, i)
            else:
                plt.plot(x, y, 'r')
                pass
        inclination = np.delete(inclination, d_point)
        print(inclination.shape[0])
        return inclination.shape[0]

    def median_filter(self, image):
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
                imageValue1[i, j] = np.sort(value.reshape(9))[4]
        imageValue1 = np.delete(imageValue1, width + 1, axis=1)
        imageValue1 = np.delete(imageValue1, height + 1, axis=0)
        imageValue1 = np.delete(imageValue1, 0, axis=1)
        imageValue1 = np.delete(imageValue1, 0, axis=0)
        return imageValue1

    def dct2(self,image):
        M = image.shape[0]
        N = image.shape[1]
        a = np.zeros([M, N], float)
        b = np.zeros([M, N], float)
        for i in range(M):
            a[i, :] = dct(image[i, :])
        for j in range(N):
            b[:, j] = dct(a[:, j])
        return b
    def dct_x(self,image):
        M = image.shape[0]
        N = image.shape[1]
        a = np.zeros([M, N], float)

        for i in range(M):
            a[i, :] = dct(image[i, :])
        return a
    def dct_y(self,image):
        M = image.shape[0]
        N = image.shape[1]
        a = np.zeros([M, N], float)
        for i in range(N):
            a[:, i] = dct(image[:, i])
        return a
    def fft_x(self,image):
        M = image.shape[0]
        N = image.shape[1]
        a = np.zeros([M, N], np.complex128)

        for i in range(M):
            a[i, :] = fft(image[i, :])
        return a
    def fft_y(self,image):
        M = image.shape[0]
        N = image.shape[1]
        a = np.zeros([M, N], np.complex128)
        for i in range(N):
            a[:, i] = fft(image[:, i])
        return a
    def Gaussian_filter(self, image, sigma):
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        x, y = np.meshgrid(x, y)
        Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
        height = image.shape[0]
        width = image.shape[1]
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        imageValue = np.zeros([height + 2, width + 2])
        imageValue1 = np.zeros([height + 2, width + 2])
        imageValue[y + 1, x] = image[y, x]
        imageValue[y + 1, x + 2] = image[y, x]
        imageValue[0] = imageValue[1]
        imageValue[-1] = imageValue[-2]
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                value = Mask * imageValue[i - 1:i + 2, j - 1:j + 2]
                imageValue1[i, j] = np.sum(value)
        imageValue1 = np.delete(imageValue1, width + 1, axis=1)
        imageValue1 = np.delete(imageValue1, height + 1, axis=0)
        imageValue1 = np.delete(imageValue1, 0, axis=1)
        imageValue1 = np.delete(imageValue1, 0, axis=0)
        return imageValue1

    def Fourier_transform(self, image, M, N):
        F = np.copy(image)
        pi = np.pi
        x = np.arange(M)
        y = np.arange(N)
        x, y = np.meshgrid(x, y)
        value1 = np.exp(-2j * pi * x / M)
        value2 = np.exp(-2j * pi * y / N)
        CT = np.zeros([M, M, N])
        CT = CT.astype(np.complex128)
        ST = np.zeros([N, N, M])
        ST = ST.astype(np.complex128)
        imsift = image.astype(np.complex128)
        for u in range(M):
            CT[u] = value1 ** u * F
        for v in range(N):
            ST[v] = value2 ** v
        for i in range(M):
            for j in range(N):
                imsift[i, j] = np.sum(CT[j] * ST[i])  # meshgrid 사용시 메모리 에러
        return imsift



    def standard_deviation_filter(self, image):
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
                m=np.mean(value)
                V=np.mean((value-m)**2)
                sigma=np.sqrt(V)
                imageValue1[i, j] = sigma
        imageValue1 = np.delete(imageValue1, width + 1, axis=1)
        imageValue1 = np.delete(imageValue1, height + 1, axis=0)
        imageValue1 = np.delete(imageValue1, 0, axis=1)
        imageValue1 = np.delete(imageValue1, 0, axis=0)
        return imageValue1
    def Gabor_Texture_Filter(self, image, lamda, theta, psi,sigma, gamma):
        # lambda — 정현파 성분의 파장
        # theta — Gabor 기능의 평행 줄무늬에 대한 법선 방향
        # psi — 사인파 함수의 위상 오프셋
        # sigma — 가우스 엔벨로프의 시그마 / 표준 편차
        # gamma — 공간 종횡비이며 가보 함수 지원의 타원도를 지정
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        x = np.arange(-1,2)
        y = np.arange(-1,2)
        x, y = np.meshgrid(x, y)
        x_dot = x * np.cos(theta) + y * np.cos(theta)
        y_dot = -x * np.sin(theta) + y * np.cos(theta)
        Mask = np.exp(-.5 * (x_dot** 2 / sigma_x ** 2 + y_dot ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / lamda * x_dot + psi)
        height = image.shape[0]
        width = image.shape[1]
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        imageValue = np.zeros([height + 2, width + 2])
        imageValue1 = np.zeros([height + 2, width + 2])
        imageValue[y + 1, x] = image[y, x]
        imageValue[y + 1, x + 2] = image[y, x]
        imageValue[0] = imageValue[1]
        imageValue[-1] = imageValue[-2]
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                value = Mask * imageValue[i - 1:i + 2, j - 1:j + 2]
                imageValue1[i, j] = np.sum(value)
        imageValue1 = np.delete(imageValue1, width + 1, axis=1)
        imageValue1 = np.delete(imageValue1, height + 1, axis=0)
        imageValue1 = np.delete(imageValue1, 0, axis=1)
        imageValue1 = np.delete(imageValue1, 0, axis=0)
        imageValue1 = imageValue1 - np.min(imageValue1)
        return imageValue1 * (255 / np.max(imageValue1))
    def Gabor_Texture_Filter1(self, image, lamda, theta, psi,sigma, gamma):
        # lambda — 정현파 성분의 파장
        # theta — Gabor 기능의 평행 줄무늬에 대한 법선 방향
        # psi — 사인파 함수의 위상 오프셋
        # sigma — 가우스 엔벨로프의 시그마 / 표준 편차
        # gamma — 공간 종횡비이며 가보 함수 지원의 타원도를 지정
        sn=1
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        height = image.shape[0]
        width = image.shape[1]
        h_x = height // sn
        w_x = width // sn
        x = np.arange(-width//(2*sn),width//(2*sn))
        y = np.arange(-height//(2*sn),height//(2*sn))
        x, y = np.meshgrid(x, y)
        x_dot = x * np.cos(theta) + y * np.cos(theta)
        y_dot = -x * np.sin(theta) + y * np.cos(theta)
        Mask = np.exp(-.5 * (x_dot** 2 / sigma_x ** 2 + y_dot ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / lamda * x_dot + psi)
        for i in range(1, sn+1):
            for j in range(1, sn+1):
                image[(i-1)*h_x:i*h_x,(j-1)*w_x:j*w_x]=image[(i-1)*h_x:i*h_x,(j-1)*w_x:j*w_x]*Mask
        return image