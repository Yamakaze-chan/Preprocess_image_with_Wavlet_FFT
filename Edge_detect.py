# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img = cv2.imread("moutain_noise.jpg", 0)

# f = np.fft.fftshift(np.fft.fft2(img))

# rows, cols = img.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# mask = np.ones((rows, cols), np.uint8)
# r = 60
# mask[crow - r : crow + r, ccol - r : ccol + r] = 0

# f_filtered = f * mask

# img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
# img_filtered = cv2.convertScaleAbs(img_filtered)

# plt.subplot(121), plt.imshow(img, cmap="gray")
# plt.title("Original Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img_filtered, cmap="gray")
# plt.title("Filtered Image"), plt.xticks([]), plt.yticks([])
# plt.show()

import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("6933_01.jpg", 0)  # load an image

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
mask[mask_area] = 0

# apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.axis("off")
ax1.imshow(img, cmap="gray")
ax1.title.set_text("Input Image")
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.imshow(magnitude_spectrum, cmap="gray")
# ax2.title.set_text("FFT of image")
# ax3 = fig.add_subplot(2, 2, 3)
# ax3.imshow(fshift_mask_mag, cmap="gray")
# ax3.title.set_text("FFT + Mask")
ax4 = fig.add_subplot(2, 2, 2)
ax4.axis("off")
ax4.imshow(img_back, cmap="gray")
ax4.title.set_text("After inverse FFT")
plt.show()
