from PyQt5.QtWidgets import *
from MyMainWindow import myMainWindow
import sys

def main():
    app = QApplication(sys.argv)
    video_gui = myMainWindow()
    video_gui.setWindowTitle("DeepFake Detection - 深度伪造检测")
    video_gui.setFixedSize(video_gui.width(), video_gui.height())
    video_gui.show()
    sys.exit(app.exec_())

main()