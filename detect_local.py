from PyQt5.QtWidgets import *
import argparse
import sys
import cv2
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
#from PyQt5.uic.properties import QtGui

from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.detect_ui import Ui_MainWindow # 导入detect_ui的界面
from models.experimental import attempt_load
from utils.datasets import letterbox

from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.plots import plot_one_box2

class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        #self.timer_video = QtCore.QTimer() # 创建定时器
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_slots()
       # self.cap = cv2.VideoCapture()
        #self.num_stop = 1 # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.output_folder = 'output/'
        self.vid_writer = None
        self.QtImg = None


        # 权重初始文件名
        #self.openfile_name_model = None
        self.openfile_name_model="weights/best.pt"

        self.model_init()
        self.center()
# 控件绑定相关操作
    def init_slots(self):
        self.ui.pushButton_img.clicked.connect(self.button_image_open)
        self.ui.pushButton_download.clicked.connect(self.button_image_download)

        #self.textBrowser
    #窗口居中
    def center(self):
        # 获取屏幕坐标
        screen = QtWidgets.QDesktopWidget().screenGeometry();
        # 获取窗口尺寸
        size = self.geometry()
        newleft=(screen.width() - size.width())/2
        newtop = (screen.height() - size.height() - 60) / 2
        self.move(newleft,newtop)
    # 加载相关参数，并初始化模型
    def model_init(self):
        # 模型相关参数配置
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        #check_requirements(exclude=('pycocotools', 'thop')) #暂时不用
        print("self.opt:", self.opt)
        # 默认使用opt中的设置（权重等）来对模型进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        print("self.opt.weights:", self.opt.weights)
        print("self.opt.save_txt:", self.opt.save_txt)
        print("weights:", weights)
        print("self.opt.view_img:", self.opt.view_img)
        # 若openfile_name_model不为空，则使用此权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model
            print("Using button choose model")

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True
        print("weights1:", weights)
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("model initial done")
        # print("self.model:", self.model)
        print("self.imgsz:", self.imgsz)
        # 设置提示框
        #QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成", buttons=QtWidgets.QMessageBox.Ok,
        #                                  defaultButton=QtWidgets.QMessageBox.Ok)

    # 目标检测
    def detect(self, name_list, img):
        '''
        :param name_list: 文件名列表
        :param img: 待检测图片
        :return: info_show:检测输出的文字信息
        '''
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            info_show = ""
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        single_info = plot_one_box2(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                    line_thickness=10)#改为3 line_thickness 为bbox的粗细大小
                        #plot_one_box(xyxy, showimg, label=label, color=colors[int(cls)], line_thickness=3)


                        info_show = info_show + single_info + "\n"
        return info_show

    def button_image_open(self):
        print('button_image_open')
        name_list = []
        try:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/images", "*.jpg;;*.png;;*.jepg;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确'+ str(reason))
        else:
            # 判断图片是否为空
            if not img_name:
                print('img name is null')
            else:
                img = cv2.imread(img_name)
                print("img_name:", img_name)
                #print("img_:", img)
                #jpg = QtGui.QPixmap(img_name).scaled(self.label_old.width(), self.label_old.height())
                #print("jpg:", jpg)
                #self.label_old.setPixmap(img)
                #self.ui.label_old.setPixmap(QPixmap(img_name))
                #self.ui.label_old.setScaledContents(True)  # 设置图像自适应界面大小

                info_show = self.detect(name_list, img)
                print("info_show", info_show)
                # 获取当前系统时间，作为img文件名
                now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                file_extension = img_name.split('.')[-1]
                new_filename = now + '.' + file_extension  # 获得文件后缀名
                file_path = self.output_folder + 'img_output/' + new_filename
                cv2.imwrite(file_path, img)
                print("file_path", file_path)
                # 检测信息显示在界面
                self.ui.textBrowser.setText(info_show)

                # 检测结果显示在界面
                self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                          QtGui.QImage.Format_RGB32)
                self.ui.label_new.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.ui.label_new.setScaledContents(True)  # 设置图像自适应界面大小

                #self.ui.label_old.setPixmap()

    def button_image_download(self):
        if self.QtImg:
            #print(self.QtImg);
            fdir, ftype = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image",
                                                      "./", "Image Files (*.jpg)")
            self.QtImg.save(fdir)
            # 设置提示框
            if fdir:
                QtWidgets.QMessageBox.information(self, u"提示", u"下载完成，图片存放在"+fdir, buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.warning(self, u"警告", u"请先检测图片", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok);
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())