from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from ui.test_ui1 import Ui_Form

class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        # self.timer_video = QtCore.QTimer() # 创建定时器
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        #self.setupUi(self)
        self.ui.pushButton.clicked.connect(self.button_image_open)

    def button_image_open(self):
        print("haha")
if __name__ == '__main__':
    #app=QApplication(sys.argv)
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window();
    current_ui.show()
    sys.exit(app.exec_())

    #current_ui = MyWindow()
    #current_ui.show()
    #sys.exit(app.exec_())
