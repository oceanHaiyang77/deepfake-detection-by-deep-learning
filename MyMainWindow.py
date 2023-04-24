import os
import re
import time
import torch
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from training.zoo.classifiers import DeepFakeClassifier

from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
# from PyQt5.QtCore import pyqtSignal
# from PyQt5.QtGui import QMovie
from GUI import Ui_MainWindow

# 在计算时弹出一个等待窗口
# class LoadingProgress(QDialog):
#     update_signal = pyqtSignal(bool)
#
#     def __init__(self, parent=None):
#         super(LoadingProgress, self).__init__(parent)
#         self.value = 0
#         self.update_signal.connect(self.update_progress)
#         vbox = QVBoxLayout(self)
#         self.steps = [f"连接服务器中...",
#                       "发送数据中...",
#                       "接收数据中...",
#                       "解析数据中..."]
#         self.movie_label = QLabel()
#         self.movie = QMovie("source/waiting.gif")
#         self.movie_label.setMovie(self.movie)
#         self.movie.start()
#         self.progress_label = QLabel()
#         self.label_update()
#
#         vbox.addWidget(self.movie_label)
#         vbox.addWidget(self.progress_label)
#         self.setLayout(vbox)
#         # self.exec_()
#
#     def label_update(self):
#         self.progress_label.setText(self.steps[self.value])
#
#     def update_progress(self, boolean: bool) -> None:
#         self.value += 1
#         if boolean and self.value < len(self.steps):
#             self.label_update()
#         else:
#             self.close()


class myMainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()

        self.model_state = 0
        self.models = []
        self.dir = ''
        self.file_state = 0

        self.setupUi(self)
        self.sld_video_pressed=False  #判断当前进度条识别否被鼠标点击
        # self.videoFullScreen = False   # 判断当前widget是否全屏
        # self.videoFullScreenWidget = myVideoWidget()   # 创建一个全屏的widget
        self.player = QMediaPlayer()
        # self.player.setLoopCount(-1)
        self.player.setVideoOutput(self.wgt_video)  # 视频播放输出的widget，就是上面定义的
        # self.btn_open.clicked.connect(self.openVideoFile)   # 打开视频文件按钮
        self.actionopen.triggered.connect(self.openVideoFile)
        self.btn_play.clicked.connect(self.playVideo)       # play
        self.player.positionChanged.connect(self.changeSlide)      # change Slide
        # self.videoFullScreenWidget.doubleClickedItem.connect(self.videoDoubleClicked)  #双击响应
        # self.wgt_video.doubleClickedItem.connect(self.videoDoubleClicked)   #双击响应
        self.sld_video.setTracking(False)
        self.sld_video.sliderReleased.connect(self.releaseSlider)
        self.sld_video.sliderPressed.connect(self.pressSlider)
        self.sld_video.sliderMoved.connect(self.moveSlider)   # 进度条拖拽跳转
        self.sld_video.ClickedValue.connect(self.clickedSlider)  # 进度条点击跳转
        self.sld_audio.valueChanged.connect(self.volumeChange)  # 控制声音播放
        self.actionload_model.triggered.connect(self.loadModel)
        self.actioncalculate.triggered.connect(self.calculate)
        self.actionabout_us.triggered.connect(self.aboutUs)
        self.btn_clear.clicked.connect(self.textClear)

    def openVideoFile(self):        # 打开文件
        file_url = QFileDialog.getOpenFileUrl(self)
        str_file_url = file_url[0].toString()[8:]
        # print("len:{}".format(len(str_file_url)))
        if len(str_file_url) == 0:
            return

        self.player.setMedia(QMediaContent(file_url[0]))  # 指定播放器的输入
        self.labelFileName.setText(str_file_url)
        self.player.play()
        self.dir = str_file_url
        self.file_state = 1
        self.textBrowser.append("Successfully load the file:")
        self.textBrowser.append(str_file_url)
        self.label_9.setText("0.00")
        self.label_3.setText("0")

    def loadModel(self):        # 导入模型
        if self.file_state == 0:
            QMessageBox.information(self, "warning", "File not loaded yet!", QMessageBox.Yes, QMessageBox.Yes)
            return
        # 模型参数文件目录
        args_weights_dir = "weights"
        args_models = ["final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
                       "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
                       "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
                       "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
                       "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
                       "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
                       "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"]
        # 设置模型参数
        self.textBrowser.append("Loading model:")
        QApplication.processEvents()
        model_paths = [os.path.join(args_weights_dir, model) for model in args_models]
        i = 1
        for path in model_paths:
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
            self.textBrowser.append("  load layer {}/7 finished".format(i))
            QApplication.processEvents()
            i += 1
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
            model.eval()
            del checkpoint
            self.models.append(model.float())

        self.model_state = 1
        self.textBrowser.append("Successfully load the model.")

    def calculate(self):        # 计算可能性
        # 检测文件和模型状态
        if self.file_state == 0:
            QMessageBox.information(self, "warning", "file not loaded yet!", QMessageBox.Yes, QMessageBox.Yes)
            return
        if self.model_state == 0:
            QMessageBox.information(self, "warning", "model not loaded yet!", QMessageBox.Yes, QMessageBox.Yes)
            return

        # 设置视频读取格式
        frames_per_video = 32
        video_reader = VideoReader()
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn)
        input_size = 380
        strategy = confident_strategy

        # 计算开始前的准备
        self.textBrowser.append("Predicting video:{}".format(self.dir))
        QApplication.processEvents()
        stime = time.time()

        # 开始计算
        predict = predict_on_video(face_extractor=face_extractor, video_path=self.dir, input_size=input_size,
                                   models=self.models, batch_size=frames_per_video, strategy=strategy,
                                   apply_compression=False)
        # time.sleep(5.0)
        # predict = 0.573566148

        # 计算完成
        elapsed = time.time() - stime
        self.textBrowser.append("Predict finished:")
        self.textBrowser.append("  confidence:{}".format(predict))
        self.textBrowser.append("  elapsed:{}".format(str(elapsed)))
        self.label_9.setText(str(elapsed))
        self.label_3.setText(str(predict))

        QMessageBox.information(self, "notice", "Predict finished.", QMessageBox.Yes, QMessageBox.Yes)

    def volumeChange(self, position):       # 调节音量
        volume = round(position/self.sld_audio.maximum()*100)
        print("音量 %f" %volume)
        self.player.setVolume(volume)
        self.lab_audio.setText("volume:"+str(volume)+"%")

    def clickedSlider(self, position):      # 点击进度条
        # 开始播放后才允许进行跳转
        if self.player.state() > 0:
            video_position = int((position / 100) * self.player.duration())
            self.player.setPosition(video_position)
            self.lab_video.setText("%.2f%%" % position)
        else:
            self.sld_video.setValue(0)

    def moveSlider(self, position):     # 拉动进度条
        self.sld_video_pressed = True
        # 开始播放后才允许进行跳转
        if self.player.duration() > 0:
            video_position = int((position / 100) * self.player.duration())
            self.player.setPosition(video_position)
            self.lab_video.setText("%.2f%%" % position)

    def pressSlider(self):
        self.sld_video_pressed = True
        print("pressed")

    def releaseSlider(self):
        self.sld_video_pressed = False

    def changeSlide(self, position):
        # 进度条被鼠标点击时不更新
        if not self.sld_video_pressed:
            self.vidoeLength = self.player.duration()+0.1
            self.sld_video.setValue(round((position/self.vidoeLength)*100))
            self.lab_video.setText("%.2f%%" % ((position/self.vidoeLength)*100))

    def playVideo(self):        # 切换 播放/暂停 状态
        if self.player.state() == 1:
            self.player.pause()
        else:
            self.player.play()

    def aboutUs(self):      # 打印“关于”信息
        about_msg = "This program is used for DeepFake Detection. " \
                    "The fake_confidence is a number from 0 to 1. " \
                    "The closer the score is to 1, the more likely the video is fake"
        self.textBrowser.append(about_msg)

    def textClear(self):
        self.textBrowser.clear()


    # def videoDoubleClicked(self, text):
    #
    #     if self.player.duration() > 0:  # 开始播放后才允许进行全屏操作
    #         if self.videoFullScreen:
    #             self.player.setVideoOutput(self.wgt_video)
    #             self.videoFullScreenWidget.hide()
    #             self.videoFullScreen = False
    #         else:
    #             self.videoFullScreenWidget.show()
    #             self.player.setVideoOutput(self.videoFullScreenWidget)
    #             self.videoFullScreenWidget.setFullScreen(True)
    #             self.videoFullScreen = True

