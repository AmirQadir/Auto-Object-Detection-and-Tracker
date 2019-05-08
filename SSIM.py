import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim


#img = img_as_float(data.camera())
img = img_as_float(cv.imread('nabeel.jpg',0)) 
rows, cols = img.shape



def mse(x, y):
    return np.linalg.norm(x - y)

img_noise = img_as_float(cv.imread('nabeel_same_size.jpg',0))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mse(img, img)
ssim_none = ssim(img, img, data_range=img.max() - img.min())

mse_noise = mse(img, img_noise)
ssim_noise = ssim(img, img_noise,
                  data_range=img_noise.max() - img_noise.min())

label = 'MSE: {:.2f}, SSIM: {:.2f}'

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(label.format(mse_none, ssim_none))
ax[0].set_title('Original image')

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
ax[1].set_title('Image with noise')


plt.tight_layout()
#plt.show()
plt.savefig('dataset/ssim.jpg')
