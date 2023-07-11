import numpy as np
import cv2

image_path = "6933_01.jpg"

# Denoise
# # Read the image
# img = mpimg.imread(image_path)

# # Convert the image to grayscale
# gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# # Compute the Fast Fourier Transform of the image
# f = fftpack.fft2(gray_img)

# # Shift the zero-frequency component to the center of the spectrum
# fshift = fftpack.fftshift(f)

# # Create a mask to filter out high-frequency noise
# rows, cols = gray_img.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# r = 60
# mask = np.zeros((rows, cols))
# mask[crow - r : crow + r, ccol - r : ccol + r] = 1

# # Apply the mask to the shifted spectrum
# fshift_filtered = fshift * mask

# # Shift the zero-frequency component back to the upper-left corner of the spectrum
# f_filtered = fftpack.ifftshift(fshift_filtered)

# # Compute the inverse Fast Fourier Transform of the filtered spectrum
# img_filtered = np.abs(fftpack.ifft2(f_filtered))

# # Display the original and denoised images
# plt.subplot(121), plt.imshow(img), plt.title("Original Image")
# plt.xticks([]), plt.yticks([])

# cv2.imwrite("output.jpg", img_filtered)
# plt.subplot(122), plt.imshow(img_filtered, cmap=plt.cm.gray), plt.title(
#     "Filtered Image"
# )

# plt.xticks([]), plt.yticks([])
# plt.show()

# Đọc hình ảnh
img = cv2.imread(image_path)

# Chuyển đổi sang grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Áp dụng biến đổi Fourier
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

# Tạo bộ lọc Butterworth trên không gian tần số
rows, cols = gray.shape
crow, ccol = int(rows / 2), int(cols / 2)
order = 1  # Độ dốc của bộ lọc
D0 = 40  # Tần số cắt
H = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        H[i, j] = 1 / (1 + (D / D0) ** (2 * order))

# Áp dụng bộ lọc Butterworth trên không gian tần số
fshift_denoised = fshift * H

# Áp dụng ngược lại biến đổi Fourier
ishift = np.fft.ifftshift(fshift_denoised)
iimg = np.fft.ifft2(ishift)
denoised = np.abs(iimg)

# Hiển thị hình ảnh gốc và hình ảnh sau khi denoise
cv2.imshow("Original Image", gray)
cv2.imshow("Denoised Image", denoised.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
