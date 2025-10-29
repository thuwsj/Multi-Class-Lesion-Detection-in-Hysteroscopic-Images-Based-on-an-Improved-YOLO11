import sys
import argparse
import cv2
import os
import time
from ultralytics import YOLO
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize

# 解析检测参数
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default=r"runs/weights/5_regnety_msam_wiou/weights/best.pt",
                    type=str, help='模型权重路径')
parser.add_argument('--conf_thre', type=float, default=0.3, help='置信度阈值')
parser.add_argument('--iou_thre', type=float, default=0.5, help='IoU阈值')
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 设置 Qt 平台插件的路径
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'


class StyleSheet:
    MAIN_WINDOW = """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #2c3e50, stop:1 #3498db);
            border-radius: 15px;
        }
    """

    BUTTON = """
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 15px;
            padding: 12px 24px;
            font-size: 14px;
            min-width: 120px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1c638e;
        }
    """

    TABLE = """
        QTableWidget {
            background-color: rgba(255, 255, 255, 200);
            border-radius: 10px;
            padding: 5px;
        }
        QHeaderView::section {
            background-color: #3498db;
            color: white;
            padding: 4px;
        }
    """

    LABEL = """
        QLabel {
            background-color: rgba(255, 255, 255, 150);
            border-radius: 10px;
            padding: 10px;
            color: #2c3e50;
        }
    """


def get_color(idx):
    colors = [
        QColor(231, 76, 60),  # 红色
        QColor(46, 204, 113),  # 绿色
        QColor(52, 152, 219),  # 蓝色
        QColor(155, 89, 182),  # 紫色
        QColor(241, 196, 15)  # 黄色
    ]
    return colors[idx % len(colors)]


class Detector(object):
    def __init__(self, weight_path, conf_threshold=0.5, iou_threshold=0.5):
        self.model = YOLO(weight_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.names = self.model.names
        print(self.names)
    def detect_image(self, img_bgr):
        results = self.model(img_bgr, verbose=True,
                             conf=self.conf_threshold,
                             iou=self.iou_threshold,
                             device=device)

        detection_data = []
        bboxes = results[0].boxes
        for idx in range(len(bboxes.cls)):
            cls_id = int(bboxes.cls[idx])
            conf = float(bboxes.conf[idx])
            box = bboxes.xyxy[idx].cpu().numpy().astype('uint32')
            label = self.names[cls_id]
            detection_data.append((label, conf, box))

            # 绘制检测框
            xmin, ymin, xmax, ymax = box
            color = get_color(cls_id)
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax),
                          (color.blue(), color.green(), color.red()), 2)
            cv2.putText(img_bgr, f'{label} {conf:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (color.blue(), color.green(), color.red()), 2)
        return img_bgr, detection_data


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能检测系统")
        self.setMinimumSize(1280, 720)
        # self.setWindowIcon(QIcon("icon.png"))  # 请准备图标文件
        self.setStyleSheet(StyleSheet.MAIN_WINDOW)

        # 初始化检测器
        self.detector = Detector(opt.weights, opt.conf_thre, opt.iou_thre)
        self.detection_history = []
        self.class_counts = {}

        # 新增定时器和摄像头相关属性
        self.video_timer = QTimer()  # 视频定时器
        self.camera_timer = QTimer()  # 摄像头定时器
        self.cap = None  # 视频捕捉对象

        # 创建主界面组件
        self.create_controls()
        self.create_layout()
        self.create_connections()

    def create_controls(self):
        # 控制按钮
        self.btn_image = QPushButton(QIcon("image_icon.png"), " 图片检测")
        self.btn_video = QPushButton(QIcon("video_icon.png"), " 视频检测")
        self.btn_camera = QPushButton(QIcon("camera_icon.png"), " 摄像头")
        self.btn_export = QPushButton(QIcon("export_icon.png"), " 导出记录")

        for btn in [self.btn_image, self.btn_video,
                    self.btn_camera, self.btn_export]:
            btn.setStyleSheet(StyleSheet.BUTTON)
            btn.setIconSize(QSize(24, 24))

        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(StyleSheet.LABEL)

        # 检测统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["类别", "数量"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setStyleSheet(StyleSheet.TABLE)

        # 检测记录表格
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["时间", "类型", "类别", "置信度", "位置"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setStyleSheet(StyleSheet.TABLE)

    def create_layout(self):
        # 左侧控制面板
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.btn_image)
        control_layout.addWidget(self.btn_video)
        control_layout.addWidget(self.btn_camera)
        control_layout.addWidget(self.btn_export)
        control_layout.addStretch()

        control_panel = QWidget()
        control_panel.setFixedWidth(200)
        control_panel.setLayout(control_layout)

        # 右侧主区域
        tab_widget = QTabWidget()

        # 统计面板
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.stats_table)
        stats_tab.setLayout(stats_layout)

        # 历史记录面板
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        history_layout.addWidget(self.history_table)
        history_tab.setLayout(history_layout)

        tab_widget.addTab(self.image_label, "实时检测")
        tab_widget.addTab(stats_tab, "统计信息")
        tab_widget.addTab(history_tab, "检测记录")

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(control_panel)
        main_layout.addWidget(tab_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_connections(self):
        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.toggle_video)
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_export.clicked.connect(self.export_records)
        self.video_timer.timeout.connect(self.process_video)  # 视频定时器连接
        self.camera_timer.timeout.connect(self.process_camera)  # 摄像头定时器连接

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.png)")
        if file_name:
            img_bgr = cv2.imread(file_name)
            img_bgr, detections = self.detector.detect_image(img_bgr)
            self.update_ui(img_bgr, detections, "图片检测")

    def toggle_video(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.btn_video.setText(" 视频检测")
            if self.cap: self.cap.release()
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi)")
            if file_name:
                self.cap = cv2.VideoCapture(file_name)
                self.video_timer.start(30)
                self.btn_video.setText(" 停止检测")

    def toggle_camera(self):
        if self.camera_timer.isActive():
            self.camera_timer.stop()
            self.btn_camera.setText(" 摄像头")
            if self.cap: self.cap.release()
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_timer.start(30)
            self.btn_camera.setText(" 停止摄像头")

    def process_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame, detections = self.detector.detect_image(frame)
            self.update_ui(frame, detections, "视频检测")

    def process_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame, detections = self.detector.detect_image(frame)
            self.update_ui(frame, detections, "摄像头检测")

    def update_ui(self, frame, detections, detect_type):
        # 更新图像显示
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新统计信息
        self.update_stats(detections)

        # 添加检测记录
        self.update_history(detections, detect_type)

    def update_stats(self, detections):
        self.class_counts.clear()
        for detection in detections:
            label = detection[0]
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

        self.stats_table.setRowCount(len(self.class_counts))
        for row, (label, count) in enumerate(self.class_counts.items()):
            color_item = QTableWidgetItem()
            color_item.setBackground(get_color(row))

            self.stats_table.setItem(row, 0, QTableWidgetItem(label))
            self.stats_table.setItem(row, 1, QTableWidgetItem(str(count)))
            # self.stats_table.setItem(row, 2, color_item)

    def update_history(self, detections, detect_type):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for detection in detections:
            label, conf, box = detection
            position = f"({box[0]},{box[1]})-({box[2]},{box[3]})"

            row = self.history_table.rowCount()
            self.history_table.insertRow(row)

            self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
            self.history_table.setItem(row, 1, QTableWidgetItem(detect_type))
            self.history_table.setItem(row, 2, QTableWidgetItem(label))
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{conf:.2f}"))
            self.history_table.setItem(row, 4, QTableWidgetItem(position))

    def export_records(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "导出记录", "", "CSV文件 (*.csv)")
        if file_name:
            with open(file_name, 'w') as f:
                f.write("时间,类型,类别,置信度,位置\n")
                for row in range(self.history_table.rowCount()):
                    items = [self.history_table.item(row, col).text()
                             for col in range(self.history_table.columnCount())]
                    f.write(",".join(items) + "\n")

    def closeEvent(self, event):
        if self.cap: self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
