from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from ui.test_ui import Ui_TestWindow

class MyWindow(QMainWindow , Ui_TestWindow):
    def __int__(self):
        super().__int__(parent=None)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.button_image_open)

    def button_image_open(self):
        print("haha")
if __name__ == '__main__':
    app=QApplication(sys.argv)
    mywindow=MyWindow();
    mywindow.show()
    sys.exit(app.exec_())
