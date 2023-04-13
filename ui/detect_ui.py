# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detect_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1309, 695)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui/img_ui/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(900, 0))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setKerning(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(1280, 630))
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_new = QtWidgets.QLabel(self.groupBox)
        self.label_new.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_new.sizePolicy().hasHeightForWidth())
        self.label_new.setSizePolicy(sizePolicy)
        self.label_new.setMinimumSize(QtCore.QSize(1000, 0))
        self.label_new.setStyleSheet("QLabel {\n"
"    \n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(204, 204, 204);\n"
"    border: 1px solid black;\n"
"}")
        self.label_new.setAlignment(QtCore.Qt.AlignCenter)
        self.label_new.setObjectName("label_new")
        self.horizontalLayout_2.addWidget(self.label_new)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setMinimumSize(QtCore.QSize(0, 0))
        self.textBrowser.setMaximumSize(QtCore.QSize(260, 16777215))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.textBrowser.setFont(font)
        self.textBrowser.setStyleSheet("QTextBrowser{\n"
"border-width: 1px;border-style: solid;\n"
"border-color: rgb(0, 0, 0)\n"
"}")
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_img = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_img.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_img.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.horizontalLayout.addWidget(self.pushButton_img)
        self.pushButton_download = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_download.setMinimumSize(QtCore.QSize(0, 50))
        self.pushButton_download.setObjectName("pushButton_download")
        self.horizontalLayout.addWidget(self.pushButton_download)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "目标检测工具"))
        self.label.setText(_translate("MainWindow", "欢迎使用目标检测工具"))
        self.groupBox.setTitle(_translate("MainWindow", "检测区"))
        self.label_new.setText(_translate("MainWindow", "请选择要检测的图片"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_download.setText(_translate("MainWindow", "下载图片"))
