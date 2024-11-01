import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from keras.models import load_model

class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        try:
            self.model = load_model('model1_catsVSdogs_10epoch.h5')  # Tải mô hình đã lưu
        except Exception as e:
            self.result_label.setText(f"Lỗi: {str(e)}")

    def initUI(self):
        self.setWindowTitle('Phân loại ảnh Mèo và Chó')
        self.setGeometry(100, 100, 800, 600)

        # Layout chính
        layout = QVBoxLayout()

        # Nhãn hiển thị ảnh
        self.image_label = QLabel(self)
        self.image_label.setText('Chọn một bức ảnh để phân loại')
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Nút chọn ảnh
        self.btn_load = QPushButton('Chọn ảnh', self)
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)

        # Nút dự đoán
        self.btn_predict = QPushButton('Dự đoán', self)
        self.btn_predict.clicked.connect(self.predict_image)
        layout.addWidget(self.btn_predict)

        # Nhãn hiển thị kết quả
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        layout.setSpacing(15)  # Thêm khoảng cách giữa các thành phần
        self.setLayout(layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.display_image(file_name)
            self.result_label.clear()  # Xóa kết quả trước khi chọn ảnh mới

    def display_image(self, file_name):
        self.image_label.setPixmap(QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio))
        self.image_path = file_name  # Lưu đường dẫn ảnh đã chọn

    def predict_image(self):
        if hasattr(self, 'image_path'):
            try:
                # Xử lý ảnh và dự đoán
                img = Image.open(self.image_path).resize((128, 128))
                img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                pred = np.argmax(self.model.predict(img_array), axis=-1)[0]
                result = 'Chó' if pred == 1 else 'Mèo'
                self.result_label.setText(f'Dự đoán: {result}')
            except Exception as e:
                self.result_label.setText(f"Lỗi khi dự đoán: {str(e)}")
        else:
            self.result_label.setText("Vui lòng chọn ảnh trước khi dự đoán")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    classifier = ImageClassifier()
    classifier.show()
    sys.exit(app.exec_())
