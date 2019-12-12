import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import PIL.Image as pilimg
im = pilimg.open('I.png')
imsi= np.array(im)
def get_image():
    delta = 0.25
    x = y = np.arange(600, 800, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1
    return Z
def do_plot(ax, Z, transform):
    im = ax.imshow(imsi, interpolation='none',origin='lower',extent=[-2, 4, -3, 2], clip_on=True)
    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
Z = get_image()
do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))
do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))
do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, .5))
do_plot(ax4, Z, mtransforms.Affine2D().rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))
plt.show()
