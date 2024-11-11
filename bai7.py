import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm đọc ảnh và chuyển đổi sang ảnh xám
def load_image(image_path):
    img = cv2.imread(image_path, 0)  # Đọc ảnh dưới dạng ảnh xám
    return img

# Làm mịn ảnh bằng bộ lọc Gaussian
def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# Phát hiện biên bằng toán tử Sobel
def sobel_edge_detection(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return sobel_combined

# Phát hiện biên bằng toán tử Prewitt
def prewitt_edge_detection(img):
    # Lấy gradient theo trục x và y
    prewitt_x = cv2.filter2D(img, cv2.CV_32F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitt_y = cv2.filter2D(img, cv2.CV_32F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))

    # Tính toán cường độ gradient
    prewitt_combined = cv2.magnitude(prewitt_x, prewitt_y)
    return prewitt_combined

# Phát hiện biên bằng toán tử Robert
def robert_edge_detection(img):
    # Tính toán gradient theo trục x và y với toán tử Roberts
    roberts_x = cv2.filter2D(img, cv2.CV_32F, np.array([[1, 0], [0, -1]]))
    roberts_y = cv2.filter2D(img, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))

    # Tính toán cường độ gradient
    roberts_combined = cv2.magnitude(roberts_x, roberts_y)
    return roberts_combined

# Phát hiện biên bằng toán tử Canny
def canny_edge_detection(img):
    return cv2.Canny(img, 100, 200)

# Tìm các đường viền để phân đoạn ảnh
def segment_image(img, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 2)
    return segmented_img

# Hàm chính để đọc ảnh, phát hiện biên và phân đoạn
def main(image_path):
    # Bước 1: Đọc ảnh và làm mịn với Gaussian
    img = load_image(image_path)
    blurred_img = gaussian_blur(img)

    # Bước 2: Phát hiện biên với các toán tử
    sobel_edges = sobel_edge_detection(blurred_img)
    prewitt_edges = prewitt_edge_detection(blurred_img)
    roberts_edges = robert_edge_detection(blurred_img)
    canny_edges = canny_edge_detection(blurred_img)

    # Bước 3: Phân đoạn ảnh từ biên của Canny
    segmented_img = segment_image(img, canny_edges)

    # Bước 4: Hiển thị kết quả
    plt.figure(figsize=(12, 8))
    plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(232), plt.imshow(sobel_edges, cmap='gray'), plt.title('Sobel Edges')
    plt.subplot(233), plt.imshow(prewitt_edges, cmap='gray'), plt.title('Prewitt Edges')
    plt.subplot(234), plt.imshow(roberts_edges, cmap='gray'), plt.title('Robert Edges')
    plt.subplot(235), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny Edges')
    plt.subplot(236), plt.imshow(segmented_img), plt.title('Segmented Image')
    plt.show()

# Chạy chương trình với đường dẫn đến ảnh vệ tinh
if __name__ == "__main__":
    main(r'D:\XuLyAnh\img.png')
