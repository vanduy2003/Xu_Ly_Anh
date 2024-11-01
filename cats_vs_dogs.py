from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dữ liệu IRIS
iris = load_iris()
X, y = iris.data, iris.target

# Tách dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng Gaussian Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
print("Accuracy of Naive Bayes:", accuracy_score(y_test, y_pred_nb))

# Sử dụng cây quyết định với Gini Index
cart_classifier = DecisionTreeClassifier(criterion="gini", random_state=42)
cart_classifier.fit(X_train, y_train)
y_pred_cart = cart_classifier.predict(X_test)
print("Accuracy of CART (Gini Index):", accuracy_score(y_test, y_pred_cart))

# Sử dụng cây quyết định với Information Gain
id3_classifier = DecisionTreeClassifier(criterion="entropy", random_state=42)
id3_classifier.fit(X_train, y_train)
y_pred_id3 = id3_classifier.predict(X_test)
print("Accuracy of ID3 (Information Gain):", accuracy_score(y_test, y_pred_id3))

# Sử dụng mạng nơ-ron (MLP)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)
y_pred_mlp = mlp_classifier.predict(X_test)
print("Accuracy of Neural Network:", accuracy_score(y_test, y_pred_mlp))


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Đường dẫn đến thư mục ảnh
data_dir = r'D:\XuLyAnh\img'

# Khởi tạo ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Tạo tập train và tập validation
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Đánh giá mô hình
val_loss, val_accuracy = model.evaluate(val_data)
print("Validation Accuracy of CNN:", val_accuracy)
