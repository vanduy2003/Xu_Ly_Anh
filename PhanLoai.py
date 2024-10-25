import time
import numpy as np
import tensorflow_datasets as tfds
from cv2 import resize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds



# 1. Tải bộ dữ liệu Flower từ tensorflow_datasets
(ds_train, ds_test), ds_info = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:]'], as_supervised=True,
                                         with_info=True)

# 2. Chuyển đổi dataset thành numpy arrays với việc resize ảnh
def preprocess_data(ds, img_size=(32, 32)):
    images = []
    labels = []
    for image, label in tfds.as_numpy(ds):
        # Chuyển đổi hình ảnh trở lại TensorFlow tensor và resize
        resized_image = resize(tf.convert_to_tensor(image), img_size)
        images.append(resized_image.numpy())  # Sau khi resize, chuyển lại thành NumPy array
        labels.append(label)
    return np.array(images), np.array(labels)

# 3. Gọi hàm để xử lý dữ liệu
X_train, y_train = preprocess_data(ds_train)
X_test, y_test = preprocess_data(ds_test)

# Kiểm tra kích thước dữ liệu sau khi xử lý
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 4. Chuẩn hóa dữ liệu
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Chuyển đổi ảnh từ 32x32x3 thành vector 1D (3072)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Chuẩn hóa dữ liệu với StandardScaler
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# 5. Khởi tạo các mô hình
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# 6. Huấn luyện và đánh giá các mô hình
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Bắt đầu tính thời gian
    start_time = time.time()

    # Huấn luyện mô hình
    model.fit(X_train_flat, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test_flat)

    # Kết thúc tính thời gian
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Tính toán các độ đo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Lưu kết quả
    results[model_name] = {
        'Time': elapsed_time,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# 7. Hiển thị kết quả
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Time: {metrics['Time']:.4f} seconds")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print("-" * 30)
