# 用于可视化分类效果展示
import sys
import threading

sys.path.append(".")
from Classifier import predict_module, initialize_model

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QDesktopWidget,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import cv2
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np
import qdarkstyle
import time
import json


class VideoClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 创建基础模型
        self.basic_model = initialize_model("model_saved\\best_model.pkl")
        self.classification_result = None

    def initUI(self):

        self.setWindowTitle("AI OR Reality")
        self.setGeometry(100, 100, 1000, 680)
        self.setWindowIcon(QIcon("utils\logo.ico"))
        self.setFont(QFont("Arial"))

        # 创建上传按钮
        self.upload_button = QPushButton("Upload video", self)
        self.upload_button.setGeometry(10, 10, 100, 30)
        self.upload_button.clicked.connect(self.upload_video)

        # 创建视频播放区域
        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 50, 485, 485)
        self.video_label.setStyleSheet("border: 2px solid black;")

        # 创建分类结果显示区域
        self.result_text = QTextEdit(self)
        self.result_text.setGeometry(10, 540, 980, 130)
        self.result_text.setReadOnly(True)
        font = QFont(
            "Times New Roman", 12, QFont.Bold
        )  # 设置字体为 Times New Roman，大小为 12，加粗
        self.result_text.setFont(font)

        # 创建分类结果图像显示区域
        self.result_image_label = QLabel(self)
        self.result_image_label.setGeometry(505, 50, 485, 485)
        self.result_image_label.setStyleSheet("border: 2px solid black;")

        # 移到中间
        self.moveCenter()

    def moveCenter(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def upload_video(self):
        # 默认文件路径
        default_path = "utils"  # 替换为你的默认文件路径

        # 使用opencv打开视频文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Please select a video file",
            default_path,
            "Video Files (*.mp4 *.avi *.mov)",
        )
        if file_path:
            self.play_video(file_path)

    def play_video(self, file_path):
        # 使用OpenCV打开视频文件
        cap = cv2.VideoCapture(file_path)

        # 循环读取视频帧
        while cap.isOpened():
            # 读取视频帧
            ret, frame = cap.read()
            # 如果视频帧读取失败，则退出循环
            if not ret:
                break
            # 如果成功读取视频帧
            else:
                # 将读取的视频帧保存为jpg格式的图片文件
                cv2.imwrite("temp_picture\\grap.jpg", frame)

            # 转换视频帧格式为RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 获取图像的高度、宽度和通道数
            h, w, ch = rgb_image.shape
            # 计算图像每行的字节数
            bytes_per_line = ch * w
            # 创建Qt图片对象
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 将Qt图片对象转换为Pixmap对象
            pixmap = QPixmap.fromImage(q_image)
            # 调整图片大小并保持宽高比
            pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
            # 在界面上显示视频帧
            self.video_label.setPixmap(pixmap)
            # 保持界面响应
            QApplication.processEvents()

            t = threading.Thread(target=self.data_predict)
            t.setDaemon(True)
            t.start()
            t.join()

            classification_result_str = json.dumps(self.classification_result, indent=4)
            # 设置 result_text 的文本内容
            self.result_text.setText(
                f"Classification result is: {classification_result_str}"
            )
            pixmap = QPixmap("temp_picture\\predicted_image.png")
            self.result_image_label.setPixmap(pixmap)
            self.result_image_label.setScaledContents(True)
            # 保持界面响应
            QApplication.processEvents()

            time.sleep(2.033)  # 视频的帧率一般是30 帧/秒（30fps）

        # 释放视频流
        cap.release()

    def data_predict(self):
        # 在此处调用模型预测函数
        self.classification_result = predict_module(["temp_picture\\grap.jpg"])

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure want to quit SIFS?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            sys.exit(0)
        else:
            if event:
                event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoClassificationApp()
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
