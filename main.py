import cv2 as cv
import sys
import numpy as np
from numpy import round
import PyQt5
from PyQt5 import QtGui, QtWidgets, QtDesigner
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QFileDialog, QPushButton, QAction
import sys
from PyQt5.QtGui import QPixmap, QIcon
from matplotlib import pyplot as plt

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()  # Call the inherited classes __init__ method
        uic.loadUi('main.ui', self)  # Load the .ui file
        self.button = self.findChild(QtWidgets.QPushButton, 'button1')
        self.button.clicked.connect(self.printButtonPressed)
        self.show()  # Show the GUI

    def printButtonPressed(self):
        # This is executed when the button is pressed
        print('printButtonPressed')

def run():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    app.exec_()
run()

def resize(image):
    image = cv.imread('')
    width = (image.shape[1] * 50 / 100)
    height = (image.shape[0] * 50 / 100)
    dsize = (int(width), int(height))
    output = cv.resize(image, dsize)
    return output


def togray(img):
    res = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(res)
    stack = np.hstack((dst, res))
    return stack


def calcHis(image):
    bgr_planes = cv.split(image)
    histSize = 256
    histRange = (0, 256)
    accumulate = False
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, histSize):
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(round(b_hist[i - 1]))),
                (bin_w * (i), hist_h - int(round(b_hist[i]))),
                (255, 0, 0), thickness=2)
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(round(g_hist[i - 1]))),
                (bin_w * (i), hist_h - int(round(g_hist[i]))),
                (0, 255, 0), thickness=2)
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(round(r_hist[i - 1]))),
                (bin_w * (i), hist_h - int(round(r_hist[i]))),
                (0, 0, 255), thickness=2)
    cv.imshow('calcHist Demo', histImage)
    cv.waitKey(0)


def linearImg(self, img):
    imgMean = np.mean(img)
    imgStd = np.std(img)
    outMean = 100
    outStd = 20
    scale = outStd / imgStd
    shift = outMean - scale * imgMean
    imgLinear = shift + scale * img

    return imgLinear

def ShowImageGray(self, image, label):
    label.clear()
    image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                         QtGui.QImage.Format_Grayscale8).rgbSwapped()
    label.setPixmap(QtGui.QPixmap.fromImage(image))

def blurImage(img):
    blur = cv.blur(img, (10, 10))
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()


def GaussblurImage(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

def MeanblurImage(img):
    median = cv.medianBlur(img,5)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(median), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

def Bilateral(img):
    blur = cv.bilateralFilter(img,9,75,75)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

# if __name__ == '__main__':
#     #run()
#     img = cv.imread('images.jpg')
#     # blurImage(img)
#     # Bilateral(img)
#     # GaussblurImage(img)
#     MeanblurImage(img)