import os
import mss
import time
import cv2 as cv
import numpy as np
from model import *
import ImageModifier
from PIL import ImageFont, ImageDraw, Image

class Game:
    def __init__(self, top, left, width, height):
        self.window = {"top":top, "left":left, "width":width, "height":height}

        self.imgMod = ImageModifier.ImageModifier()

        self.bigBoardClean = cv.resize(np.concatenate((self.__setDeck(), self.__setSuits())), (0,0), fx = 0.75, fy = 0.75).copy()
        self.bigBoard = self.bigBoardClean.copy()
        
    def __suitInterpretor(self, probabilities):
        num = np.argmax(probabilities)
        if num == 0:
            return "Diamonds"
        elif num == 1:
            return "Hearts"
        elif num == 2:
            return "Clubs"
        elif num == 3:
            return "Spades"
    
    def __suitImage(self, num):
        if num == 0:
            return "\u2666"
        elif num == 1:
            return "\u2665"
        elif num == 2:
            return "\u2663"
        elif num == 3:
            return "\u2660"
        
    def __numberInterpretor(self, num):
        if num < 9:
            return f"{num + 2}"
        elif num == 9:
            return "J"
        elif num == 10:
            return "Q"
        elif num == 11:
            return "K"
        elif num == 12:
            return "A"
        
    def __reset(self):
        self.bigBoard = self.bigBoardClean.copy()
    
    def __setDeck(self):
        ret = np.ones((10,1310, 3))
        
        for i in range(4):
            big = np.zeros((100,10,3))
            image = Image.new("RGB", (50, 100), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Arial Unicode.ttf", 50)
            unicode_char = self.__suitImage(i)
            draw.text((10, 15), unicode_char, font=font, fill= "red" if i < 2 else "black")

            image = np.asarray(image)
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            for j in range(13):
                temp = np.ones((100,50,3))
                temp = cv.putText(temp, self.__numberInterpretor(j), (30,60) if j != 8 else (10, 60), cv.FONT_HERSHEY_COMPLEX, 1, 0)
                big = np.concatenate((big, temp, image), axis = 1)
            
            ret = np.concatenate((ret, big), axis = 0)
        return ret
    
    def __detectScoreboard(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        mean1 = np.mean(img[855:955, 545:575])
        mean2 = np.mean(img[855:955, 725:755])
        mean3 = np.mean(img[855:955, 905:935])

        return mean1 == 233.0 and mean2 == 233.0 and mean3 == 233.0
    
    def __playCard(self, suit, num):
        yOff = 8 + np.argmax(suit) * 75
        xOff = 8 + np.argmax(num) * 75
        cv.line(self.bigBoard, (xOff, yOff), (xOff + 75, yOff + 75), 0, 2)
    
    def __checkStart(self, suits, numbers):
        for i in range(4):
            if np.argmax(suits[i]) == 2 and np.argmax(numbers[i]) == 0:
                return i
        return -1
    
    def __findWinner(self, lead, suits, numbers):
        leadSuit = np.argmax(suits[lead])
        leadNumber = np.argmax(numbers[lead])

        i = 0
        idx = lead
        ret = idx

        while i < 4:
            i += 1
            if np.argmax(suits[idx])  == leadSuit and np.argmax(numbers[idx]) > leadNumber:
                ret = idx
                leadNumber = np.argmax(numbers[idx])
            elif np.argmax(suits[idx])  != leadSuit:
                self.__crossSuit(idx, leadSuit)
            idx += 1
            idx %= 4
        return ret
    
    def __createSuitsForBoard(self, player):
        ret = np.ones((100,100, 3))
        ret = cv.putText(ret, f"Player {player}", (10,60), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        for i in range(4):
            image = Image.new("RGB", (100, 100), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Arial Unicode.ttf", 60)
            unicode_char = self.__suitImage(i)
            draw.text((30, 10), unicode_char, font=font, fill= "black" if i > 1 else "red")
            image = np.asarray(image)
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            ret = np.concatenate((ret, image), axis = 1)

        return ret

    def __setSuits(self):
        p0 = self.__createSuitsForBoard(0)
        p1 = self.__createSuitsForBoard(1)
        p2 = self.__createSuitsForBoard(2)
        p3 = self.__createSuitsForBoard(3)
        
        topRow = np.concatenate((np.zeros((100,405,3)), p0, np.zeros((100,405,3))), axis=1)
        middleRow = np.concatenate((p3, np.zeros((100,310,3)),p1), axis = 1)
        bottomRow = np.concatenate((np.zeros((100,405,3)), p2, np.zeros((100,405,3))), axis=1)

        ret = np.concatenate((np.zeros((10,1310,3)),topRow, middleRow, bottomRow))

        return ret
    
    def __crossSuit(self, player, suit):
        color = (0,0,255) if suit > 1 else (0,0,0)
        if player == 0:
            yOff = 315
            xOff = 379 + suit * 75
        elif player == 3:
            yOff = 390
            xOff = 75 + suit * 75
        elif player == 2:
            yOff = 465
            xOff = 379 + suit * 75
        elif player == 1:
            yOff = 390
            xOff = 683 + suit * 75
        
        cv.line(self.bigBoard, (xOff, yOff), (xOff + 75, yOff + 75), color, 2)

    def playGame(self):
        sct = mss.mss()
        
        suitModelName = "suitClassifierv2.keras"
        if(os.path.isfile(suitModelName)):
            print("loading suit model")
            modelSuit = load_model(suitModelName)
        else:
            print("Suit model not found")
            return
        
        numberModelName = "numberClassifierV2.keras"
        if(os.path.isfile(numberModelName)):
            print("loading number model")
            modelNumber = load_model(numberModelName)
        else:
            print("Number model not found")
            return
        
        trick = 0
        lead = -1
        loaded = True
        keepingCount = False
        smallCount = 0

        print("READY TO PLAY")
        while True:
            cv.imshow("Count", self.bigBoard)
            key = cv.waitKey(10)
            if key == 32:
                break

            img = sct.grab(self.window)
            img = np.array(img)

            if self.__detectScoreboard(img):
                print("Scoreboard detected")
                self.__reset()
                trick = 0
                time.sleep(8.5)
                continue

            arr = self.imgMod.parseImage(img)
            count = 0
            for i in arr:
                if np.mean(i) > 95 / 255:
                    count += 1
            
            if count == 4 and loaded:
                loaded = False
                keepingCount = True
            elif count <= 2:
                loaded = True
            
            if keepingCount == True:
                smallCount += 1

            if smallCount > 10:
                keepingCount = False
                smallCount = 0
                test = np.array(arr).reshape(4,35,25,1)
                suits = modelSuit.predict(test, verbose = 0)
                numbers = modelNumber.predict(test, verbose = 0)

                if trick == 0:
                    possibleLead = self.__checkStart(suits, numbers)
                    if possibleLead != -1:
                        #print("Moving on")
                        lead = possibleLead
                    else:
                        #print("Too soon")
                        continue
                
                for i in range(4):
                    self.__playCard(suits[i], numbers[i])

                trick += 1
                lead = self.__findWinner(lead, suits, numbers)
                print(f"Player {lead} won the trick with {self.__numberInterpretor(np.argmax(numbers[lead]))} of {self.__suitInterpretor(suits[lead])}")
