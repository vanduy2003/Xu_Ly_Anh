import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc dưới dạng grayscale
image = cv2.imread('img2.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Dò biên sử dụng toán tử Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo hướng x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo hướng y

# Kết hợp Sobel theo hai hướng
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 2. Dò biên sử dụng Laplacian of Gaussian (LoG)
# Gaussian Blur để giảm nhiễu
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Dò biên sử dụng Laplacian
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=3)

# Hiển thị kết quả
plt.figure(figsize=(10, 6))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh Gốc')

# Ảnh Sobel (Kết hợp hai hướng)
plt.subplot(1, 3, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Dò Biên Sobel')

# Ảnh LoG (Laplacian of Gaussian)
plt.subplot(1, 3, 3)
plt.imshow(laplacian, cmap='gray')
plt.title('Dò Biên Laplace of Gaussian')

plt.show()
