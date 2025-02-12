import math
import imutils
import cv2 as cv
import numpy as np

class ImageModifier:
    def __init__(self):
        pass

    def parseImage(self, img):
        height = 140
        width = 100
        imgTop = img[635:635+height, 565:565+width]
        imgBot = img[1155:1155+height, 795:795+width]
        imgBot = np.fliplr(imgBot)
        imgBot = np.flipud(imgBot)

        cardRotateLeft = imutils.rotate(img, -0.7083333333333334 * 360 / (2 * math.pi))
        cardRotateRight = imutils.rotate(img, 1.0833333333333333 * 360 / (2 * math.pi))
        imgLeft = cardRotateLeft[695:695 + height, 495:495 + width]
        imgRight = cardRotateRight[645:645 + height, 605:605 + width]

        imgBot = self.__shrinkAndGray(imgBot)
        imgTop = self.__shrinkAndGray(imgTop)
        imgLeft = self.__shrinkAndGray(imgLeft)
        imgRight = self.__shrinkAndGray(imgRight)

        return [imgTop, imgRight, imgBot, imgLeft]
    
    def __shrinkAndGray(self, img):
        img = cv.resize(img,(0,0), fx = .25, fy = .25)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img / 255
