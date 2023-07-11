import matplotlib.pyplot as plt
import skimage.io
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import cv2
import numpy as np

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2, dtype=cv2.CV_64F)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

path = "6933_01.jpg"
original = img_as_float(skimage.io.imread(path)[100:250, 50:300])
#print(original)

sigma = 0.12
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

im_bayes = denoise_wavelet(noisy, channel_axis=-1, convert2ycbcr=True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
im_visushrink = denoise_wavelet(noisy, channel_axis=-1, convert2ycbcr=True,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est/8, rescale_sigma=True)

# VisuShrink is designed to eliminate noise with high probability, but this
# results in a visually over-smooth appearance.  Repeat, specifying a reduction
# in the threshold by factors of 2 and 4.

# Compute PSNR as an indication of image quality
psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)
print(psnr_noisy, psnr_bayes, psnr_visushrink)

ax[0, 0].imshow(original)
ax[0, 0].axis('off')
ax[0, 0].set_title('Original')
ax[0, 1].imshow(noisy)
ax[0, 1].axis('off')
ax[0, 1].set_title(f'Noisy\nPSNR={psnr_noisy:0.4g}')
ax[1, 0].imshow(im_bayes)
ax[1, 0].axis('off')
ax[1, 0].set_title(f'Wavelet denoising\n(BayesShrink)\nPSNR={psnr_bayes:0.4g}')
ax[1, 1].imshow(im_visushrink)
ax[1, 1].axis('off')
ax[1, 1].set_title(
    'Wavelet denoising\n(VisuShrink, $\\sigma=\\sigma_{est}/8$)\n'
     'PSNR=%0.4g' % psnr_visushrink)

plt.show()
"""
print("MSE between noisy image and BayesShrink:" )
print("RED:\t" + str(mse(original[:,:,2], im_bayes[:,:,2])))
print("GREEN:\t" + str(mse(original[:,:,1], im_bayes[:,:,1])))
print("BLUE:\t" + str(mse(original[:,:,0], im_bayes[:,:,0])))
print("MSE between noisy image and VisuShrink")
print("RED:\t" + str(mse(original[:,:,2], im_visushrink2[:,:,2])))
print("GREEN:\t" + str(mse(original[:,:,1], im_visushrink2[:,:,1])))
print("BLUE:\t" + str(mse(original[:,:,0], im_visushrink2[:,:,0])))
"""