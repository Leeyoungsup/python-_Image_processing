import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk,Image
sigma, theta, Lambda, psi, gamma=10,0,30,0,0.25
sigma_x = sigma
sigma_y = float(sigma) / gamma 
nstds = 3
xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
xmax = np.ceil(max(1, xmax))
ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
ymax = np.ceil(max(1, ymax))
xmin = -xmax
ymin = -ymax
(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
x_theta = x * np.cos(theta) + y * np.sin(theta)
y_theta = -x * np.sin(theta) + y * np.cos(theta)
gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
plt.imshow(gb,'gray')
plt.show()
