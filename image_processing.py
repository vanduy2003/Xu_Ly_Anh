import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm hiển thị ảnh
def show_image(title, image):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Tắt trục tọa độ
    plt.show()

# Đọc ảnh đầu vào dưới dạng grayscale (ảnh xám)
image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Không thể mở được ảnh, vui lòng kiểm tra đường dẫn ảnh.")
    exit()

# Hiển thị ảnh gốc
show_image('Ảnh Gốc', image)

# 1. Ảnh âm tính
negative_image = 255 - image
show_image('Ảnh Âm Tính', negative_image)

# 2. Tăng độ tương phản (scale ảnh giữa 0 và 255)
alpha = 1.5  # Hệ số tăng tương phản
beta = 0     # Giá trị bù sáng
contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
show_image('Ảnh Tăng Độ Tương Phản', contrast_image)

# 3. Biến đổi Logarithmic
c = 255 / np.log(1 + np.max(image))  # Hệ số điều chỉnh log
log_image = c * (np.log(1 + image))  # Áp dụng biến đổi log
log_image = np.array(log_image, dtype=np.uint8)  # Chuyển về kiểu uint8
show_image('Biến Đổi Log', log_image)

# 4. Cân bằng Histogram
equalized_image = cv2.equalizeHist(image)
show_image('Cân Bằng Histogram', equalized_image)
