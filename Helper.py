import os
import mss
import math
import imutils
import cv2 as cv
import numpy as np
from model import *
from PIL import Image

#count the files of a specified path
#input string returns int
def countFiles(path):
    temp = os.listdir(path)
    return len(temp)

#converts a BGR image to grayscale
#input np array 130x100x3 returns np array 35x25
def shrinkAndGray(img):
    #img = cv.resize(img,(0,0), fx = .25, fy = .25)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

#parses the screenshot and returns a list of the 4 areas where cards are played
def parseImage(img):
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

    imgBot = shrinkAndGray(imgBot)
    imgTop = shrinkAndGray(imgTop)
    imgLeft = shrinkAndGray(imgLeft)
    imgRight = shrinkAndGray(imgRight)

    ret = [imgTop, imgBot, imgLeft, imgRight]
    return ret

def saveImage(img, i, path):
    imgPng = Image.fromarray(img)
    imgPng.save(f"{path}{i}.png")

#returns a one hot encoding for each suit
def encode(suit):
    if suit == "D":
        return [1,0,0,0]
    elif suit == "H":
        return [0,1,0,0]
    elif suit == "C":
        return [0,0,1,0]
    elif suit == "S":
        return [0,0,0,1]

#goes through saved images, shows them, and takes input from keyboard to classify the suit of the image
def classifySuits(path, to):
    fileToX = f"{to}/trainingDataX.npy"
    fileToY = f"{to}/trainingDataY.npy"
    trainingDataX = []
    trainingDataY = []
    if(os.path.isfile(fileToX) and os.path.isfile(fileToY)):
        print("File exists")
        trainingDataX = list(np.load(fileToX))
        trainingDataY = list(np.load(fileToY))
        print(f"Already done {len(trainingDataX)} imgs with {len(trainingDataY)} classifications")
    else:
        print("File doesn't exist")
        return

    beg = len((trainingDataY)) + 1
    end = countFiles(path) - 1

    print(f"starting on {beg}")
    print(f"total to do {end}")
    while beg <= end:
        img = cv.imread(f"{path}/IMG{beg}.png", cv.IMREAD_UNCHANGED)
        
        cv.imshow("IMG",img)
        key = cv.waitKey(0)
        
        encoding = []
        if key == 99:
            #C
            print("CLUBS")
            encoding = encode("C")
        elif key == 100:
            #D
            print("DIAMONDS")
            encoding = encode("D")
        elif key == 104:
            #H
            print("HEARTS")
            encoding = encode("H")
        elif key == 115:
            #S
            print("SPADES")
            encoding = encode("S")
        elif key == 113:
            #Q
            break
        
        encoding = np.array(encoding)
        
        trainingDataX.append(img)
        trainingDataY.append(encoding)
        beg += 1
    print(f"SAVING {len(trainingDataX)}")
    print(f"SAVING {len(trainingDataY)}")
    
    np.save(fileToX, trainingDataX)
    np.save(fileToY, trainingDataY)
    
def interpreteEncoding(encoding):
    if encoding[0] == 1:
        return "Diamond"
    elif encoding[1] == 1:
        return "Heart"
    elif encoding[2] == 1:
        return "Club"
    elif encoding[3] == 1:
        return "Spade"
    
#goes through the images and their encodings
def loadData(path):
    Xpath = path + "X" + ".npy"
    Ypath = path + "Y" + ".npy"
    if(os.path.isfile(Xpath) and os.path.isfile(Ypath)):
        print("File exists")
        trainingDataX = list(np.load(Xpath))
        print(f"length {len(trainingDataX)}")
        trainingDataY = list(np.load(Ypath))
        print(f"length {len(trainingDataY)}")
    else:
        print("File doesn't exist")
        return
    
    for i in range(len(trainingDataX)):
        temp = trainingDataX[i]
        add = np.zeros_like(temp)
        add = cv.putText(add, interpreteEncoding(trainingDataY[i]), (0,70), cv.FONT_HERSHEY_COMPLEX, 0.75, (255))
        fin = np.concatenate((temp,add))
        cv.imshow("IMG", fin)
        key = cv.waitKey(0)
        if key == 113:
            break

#screen records the window and saves the played cards when all 4 cards are played
def captureScreenshots(window):
    sct = mss.mss()
    fileCount = countFiles("UnlabeledData")
    loaded = True
    keepingCount = False
    smallCount = 0
    initial = fileCount
    while True:
        img = sct.grab(window)
        img = np.array(img)

        imgCopy = img.copy()
        arr = parseImage(imgCopy)
        [imgTop, imgBot, imgLeft, imgRight] = arr
        count = 0
        for i in arr:
            temp = np.mean(i)
            if temp > 95:
                count += 1
            
        if count == 4 and loaded:
            loaded = False
            keepingCount = True
        elif count <= 2:
            loaded = True

        if keepingCount == True:
            smallCount += 1

        resultVert = np.concatenate((imgTop, imgBot), axis = 1)
        resultHor = np.concatenate((imgLeft, imgRight), axis = 1)
        result = np.concatenate((resultVert, resultHor), axis = 0)

        mask = np.zeros_like(result)
        cv.putText(mask, f"{count}", (50,50),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        comb = np.concatenate((result, mask))
        cv.imshow("Focused", comb)

        if smallCount > 10:
            keepingCount = False
            smallCount = 0
            print("Screenshot here")
            saveImage(imgTop, fileCount, "UnlabeledData/IMG")
            fileCount += 1
            saveImage(imgBot, fileCount, "UnlabeledData/IMG")
            fileCount += 1
            saveImage(imgLeft, fileCount, "UnlabeledData/IMG")
            fileCount += 1
            saveImage(imgRight, fileCount, "UnlabeledData/IMG")
            fileCount += 1
            
        key = cv.waitKey(10)
        if(key == 113):
            break
        elif key != -1:
            print(f"top {np.mean(imgTop)} bot {np.mean(imgBot)} left {np.mean(imgLeft)} right{np.mean(imgRight)}")
    print(f"pictures taken {fileCount - initial}")

def getDataStatsSuits():
    fileToY = "LabeledSuits/trainingDataY.npy"
    trainingDataY = []
    if( os.path.isfile(fileToY)):
        print("File exists")
        trainingDataY = np.load(fileToY)
        
    else:
        print("File doesn't exist")
        return
    count = np.sum(trainingDataY, axis=0)
    print(count)

#Go through all the images of a certain suit
def checkClassifications(suit):
    fileToY = "LabeledSuits/trainingDataY.npy"
    trainingDataY = []
    fileToX= "LabeledSuits/trainingDataX.npy"
    trainingDataX = []
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("Files exist")
        trainingDataX = list(np.load(fileToX))
        print(f"length {len(trainingDataX)}")
        trainingDataY = list(np.load(fileToY))
        print(f"length {len(trainingDataY)}")
    else:
        print("Files don't exist")
        return
    
    for i in range(len(trainingDataX)):
        encoding = trainingDataY[i]
        if(encoding[suit] != 1):
            continue
        temp = trainingDataX[i]
        temp = cv.resize(temp, (0,0), fx = 4, fy = 4)
        add = np.zeros_like(temp)
        add = cv.putText(add, f"{i}", (0,70), cv.FONT_HERSHEY_COMPLEX, 0.75, (255))
        fin = np.concatenate((temp,add))
        cv.imshow("IMG", fin)
        key = cv.waitKey(0)
        if key == 113:
            break
        elif key == 0:
            print(f"problem on {i}")

#update any misclassifications
def updateData(i, before, after):
    fileToY = "LabeledSuits/trainingDataY.npy"
    trainingDataY = []
    fileToX= "LabeledSuits/trainingDataX.npy"
    trainingDataX = []
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        trainingDataX = list(np.load(fileToX))
        print(f"length {len(trainingDataX)}")
        trainingDataY = list(np.load(fileToY))
        print(f"length {len(trainingDataY)}")
    else:
        print("File doesn't exist")
        return
    trainingDataY[i][before] = 0
    trainingDataY[i][after] = 1
    
    np.save(fileToY, trainingDataY)

#check a certain classification
def checkData(i):
    fileToY = "LabeledSuits/trainingDataY.npy"
    trainingDataY = []
    fileToX= "LabeledSuits/trainingDataX.npy"
    trainingDataX = []
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        trainingDataX = list(np.load(fileToX))
        print(f"length {len(trainingDataX)}")
        trainingDataY = list(np.load(fileToY))
        print(f"length {len(trainingDataY)}")
    else:
        print("File doesn't exist")
        return
    
    temp = trainingDataX[i]
    temp = cv.resize(temp, (0,0), fx = .25, fy = .25)
    add = np.zeros_like(temp)
    add = cv.putText(add, interpreteEncoding(trainingDataY[i]), (0,70), cv.FONT_HERSHEY_COMPLEX, 0.5, (255))
    fin = np.concatenate((temp,add))
    cv.imshow("IMG", fin)
    cv.waitKey(0)

def resizeData():
    fileToX= "LabeledSuits/trainingDataX.npy"
    trainingDataX = []
    if(os.path.isfile(fileToX)):
        print("File exists")
        trainingDataX = list(np.load(fileToX))
        print(f"length {len(trainingDataX)}")
    else:
        print("File doesn't exist")
        return
    
    for i in range(len(trainingDataX)):
        temp = trainingDataX[i]
        temp = cv.resize(temp, (0,0), fx = .25, fy = .25)
        trainingDataX[i] = temp
    
    np.save(fileToX, trainingDataX)

def trainModelSuit():
    Xpath = "LabeledSuits/trainingDataX.npy"
    Ypath = "LabeledSuits/trainingDataY.npy"
    if(os.path.isfile(Xpath) and os.path.isfile(Ypath)):
        print("File exists")
        trainingDataX = np.load(Xpath)
        print(f"length {len(trainingDataX)}")
        trainingDataY = np.load(Ypath)
        print(f"length {len(trainingDataY)}")
    else:
        print("File doesn't exist")
        return
    
    trainingDataX = trainingDataX.reshape((len(trainingDataX), 35,25,1))
    modelName = "suitClassifierv2.keras"

    if(os.path.isfile(modelName)):
        print("loading model")
        model = loadModel(modelName)
    else:
        print("creating new model")
        model = createModelSuit()
    
    
    model.fit(trainingDataX, trainingDataY, epochs = 5, batch_size = 16)
    model.save(modelName)
    print("DONE")

def convertImg(img):
    img = cv.resize(img, (0,0), fx = .25, fy = .25)
    img = img / 255.
    img = img.reshape(1,35,25,1)
    return img

def roundPrediction(arr):
    idx = np.argmax(arr)
    arr[idx] = 1
    return arr

#test the model using live input, saves results
def liveTesting(window):
    sct = mss.mss()
    loaded = True
    keepingCount = False
    smallCount = 0
    modelName = "suitClassifierv2.keras"
    if(os.path.isfile(modelName)):
        print("loading model")
        model = loadModel(modelName)
    fileToX = "PredictedSuits/Imgs.npy"
    fileToY = "PredictedSuits/predictions.npy"
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        imgCnvs = list(np.load(fileToX))
        print(f"length {len(imgCnvs)}")
        predictions = list(np.load(fileToY))
        print(f"length {len(predictions)}")
    else:
        print("File doesn't exist")
        imgCnvs = []
        predictions = []

    testResult = np.zeros((50, 50))
    
    while(True):
        img = sct.grab(window)
        img = np.array(img)
        imgCopy = img.copy()
        arr = parseImage(imgCopy)
        [imgTop, imgBot, imgLeft, imgRight] = arr
        count = 0
        for i in arr:
            temp = np.mean(i)
            if temp > 95:
                count += 1
            
        if count == 4 and loaded:
            loaded = False
            keepingCount = True
        elif count <= 2:
            loaded = True

        if keepingCount == True:
            smallCount += 1

        resultVert = np.concatenate((imgTop, imgBot), axis = 1)
        resultHor = np.concatenate((imgLeft, imgRight), axis = 1)
        result = np.concatenate((resultVert, resultHor), axis = 0)
        mask = np.zeros_like(result)
        cv.putText(mask, f"{count}", (50,50),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        comb = np.concatenate((result, mask))
        #comb = np.concatenate((result, mask, testResult))

        cv.imshow("Focused", testResult)
        

        if smallCount > 10:
            keepingCount = False
            smallCount = 0
            testResult = comb
            imgTopCnv = convertImg(imgTop)
            imgBotCnv = convertImg(imgBot)
            imgLeftCnv = convertImg(imgLeft)
            imgRightCnv = convertImg(imgRight)

            imgCnvs.append(imgTopCnv)
            imgCnvs.append(imgBotCnv)
            imgCnvs.append(imgLeftCnv)
            imgCnvs.append(imgRightCnv)
            
            preTop = model.predict(imgTopCnv)
            preBot = model.predict(imgBotCnv)
            preLeft = model.predict(imgLeftCnv)
            preRight = model.predict(imgRightCnv)

            predictions.append(preTop[0])
            predictions.append(preBot[0])
            predictions.append(preLeft[0])
            predictions.append(preRight[0])
            
            
        key = cv.waitKey(10)
        if(key == 113):
            break
    
    np.save(fileToX, imgCnvs)
    np.save(fileToY, predictions)
    print(f"final length {len(imgCnvs)}")    
    print(f"final length {len(predictions)}")

def goThroughTrainingData():
    model = load_model("suitClassifierv2.keras")
    Xpath = "LabeledSuits/trainingDataX.npy"
    Ypath = "LabeledSuits/trainingDataY.npy"
    if(os.path.isfile(Xpath) and os.path.isfile(Ypath)):
        print("File exists")
        trainingDataX = np.load(Xpath)
        print(f"length {len(trainingDataX)}")
        trainingDataY = np.load(Ypath)
        print(f"length {len(trainingDataY)}")

    
    trainingDataX = trainingDataX / 255.0
    trainingDataX = trainingDataX.reshape((len(trainingDataX), 35,25,1))
    pre = model.predict(trainingDataX)
    #temp = temp / 255.0
    #temp = temp.reshape((440, 140, 100, 1))
    #pre = model.predict(temp)
    
    right = 0
    wrong = 0
    print("Start")
    arr = [0,0,0,0]
    for i in range(len(trainingDataX)):
        correct = interpreteEncoding(trainingDataY[i])
        predicted = interpreteEncoding(roundPrediction(pre[i]))
        arr[np.argmax(pre[i])] += 1
        if(np.argmax(pre[i]) == np.argmax(trainingDataY[i])):
            right += 1
            print(f"correctly predicted {correct}")
        else:
            wrong += 1
            print(f"incorrectly predicted {predicted} instead of {correct}")
    print(f"Final tally {right} correct {wrong} incorrect")
    print(arr)
    print("done")

def goThroughPredictions():
    fileToX = "PredictedSuits/Imgs.npy"
    fileToY = "PredictedSuits/predictions.npy"
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        imgs = np.load(fileToX)
        print(f"length {len(imgs)}")
        print(imgs.shape)
        predictions = np.load(fileToY)
        print(f"length {len(predictions)}")
        print(predictions.shape)
    else:
        print("File doesn't exist")
        return
    
    for i in range(len(imgs)):
        temp = imgs[i]
        temp = temp.reshape((35,25))
        temp = cv.resize(temp, (0,0), fx = 4, fy = 4)
        add = np.zeros_like(temp)
        add = cv.putText(add, f"{interpreteEncoding(roundPrediction(predictions[i]))}", (0,70), cv.FONT_HERSHEY_COMPLEX, 0.75, (255))
        fin = np.concatenate((temp,add))
        cv.imshow("IMG", fin)
        key = cv.waitKey(0)
        if key == 113:
            break
        elif key == 0:
            print(f"problem on {i}")
    
def moveFiles():
    fileCount = countFiles("UnlabeledData")
    print(f"Start amount {fileCount}")
    file = "PredictedSuits/Imgs.npy"
    trainingData = []
    if(os.path.isfile(file)):
        trainingData = np.load(file)
        print(f"File exists with length {len(trainingData)}")
    else:
        return
    
    for i in range(136,len(trainingData)):
        img = trainingData[i]
        img = img.reshape(35,25)
        img = img * 255
        img = cv.resize(img, (0,0), fx = 4, fy = 4)
        img = np.round(img).astype(np.uint8)
        saveImage(img, fileCount, "UnlabeledData/IMG")
        fileCount += 1
        
        
    
    fileCount = countFiles("UnlabeledData")
    print(f"Start amount {fileCount - 1}")

#manually go through predictions and save correct classifications
def classifySuitsFromPredictions():
    fileFromX = "PredictedSuits/Imgs.npy"
    predictionPath = "PredictedSuits/predictions.npy"
    fileToX = "LabeledSuits/trainingDataX.npy"
    fileToY = "LabeledSuits/trainingDataY.npy"
    X = np.load(fileFromX)
    X = X.reshape(len(X), 35, 25)
    
    trainingDataX = []
    trainingDataY = []
    X = np.load(fileFromX)

    if(os.path.isfile(predictionPath)):
        print("Predictions exist")
        predictions = np.load(predictionPath)
    if(os.path.isfile(fileToX) and os.path.isfile(fileToY)):
        print("File exists")
        trainingDataX = list(np.load(fileToX))
        trainingDataY = list(np.load(fileToY))
        print(f"Already done {len(trainingDataX)} imgs with {len(trainingDataY)} classifications")
    else:
        print("File doesn't exist")
        return
        
    
    
    beg = 535
    end = 596

    print(f"starting on {beg}")
    print(f"total to do {end}")

    i = 535
    while beg < end:
        img = X[i]
        img = img.reshape(35,25)
        big = cv.resize(img, (0,0), fx = 4, fy = 4)
        p = predictions[i]
        i += 1
        pre = np.zeros_like(big)
        pre = cv.putText(pre, interpreteEncoding(roundPrediction(p)), (0,70), cv.FONT_HERSHEY_COMPLEX, 0.75, (255))
        show = np.concatenate((big, pre))
        cv.imshow("IMG",show)

        key = cv.waitKey(0)
        
        encoding = [0,0,0,0]
        if key == 99:
            print(f"CLUBS {i}")
            encoding = encode("C")
        elif key == 100:
            print(f"DIAMONDS {i}")
            encoding = encode("D")
        elif key == 104:
            print(f"HEARTS {i}")
            encoding = encode("H")
        elif key == 115:
            print(f"SPADES {i}")
            encoding = encode("S")
        elif key == 113:
            break
        elif key == 3:
            idx = np.argmax(p)
            encoding[idx] = 1
            print(f"Correct {i}")
        
        encoding = np.array(encoding)
        
        
        #trainingDataX.append(cv.resize(img, (0,0), fx = .25, fy = .25))
        trainingDataX.append(img)
        trainingDataY.append(encoding)
        beg += 1
    
    print(f"SAVING {len(trainingDataX)}")
    print(f"SAVING {len(trainingDataY)}")
    
    np.save(fileToX, trainingDataX)
    np.save(fileToY, trainingDataY)

#used to capture live screenshots 
def viewWindow(window, highlight = False, x = 0, y = 0, w = 0, h = 0):
    sct = mss.mss()
    i = 6
    while True:
        img = sct.grab(window)
        img = np.array(img)
        imgCopy = img.copy()
        if(highlight):
            cv.rectangle(imgCopy, (x,y), (x + w, y + h), (255,0,0), 2)

        cv.imshow("Focused", imgCopy)
            
        key = cv.waitKey(10)
        if(key == 113):
            break
        elif key != -1:
            saveImage(img, i, "ImageTest")
            i += 1
            print("Saved image")
    print("DONE")

#used to determine threshold values for detecting when scoreboard is showing
def scoreboardStillTesting():
    i = 6
    count = 0
    while True and count < 1:
        img = cv.imread(f"ImageTest{i}.png", cv.IMREAD_UNCHANGED)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgCopy = img.copy()
        cv.rectangle(imgCopy, (545,805), (575, 905), (255,0,0), 2)
        cv.rectangle(imgCopy, (725,805), (755, 905), (255,0,0), 2)
        cv.rectangle(imgCopy, (905,805), (935, 905), (255,0,0), 2)
        cv.imshow("Current box", imgCopy)
        if count == 0:
            arr = img[805:905, 545:575]
            print(np.mean(arr))
            arr = img[805:905, 725:755]
            print(np.mean(arr))
            arr = img[805:905, 905:935]
            print(np.mean(arr))
        key = cv.waitKey(0)
        if(key == 113):
            break
        elif key == 44:
            #prev photo
            i -= 1
            i = max(2, i)
            count = 0
        elif key == 46:
            #next photo
            i += 1
            i = min(i, 5)
            count = 0
        elif key != -1:
            print(key)
        count += 1

def detectScoreboard(img):
    mean1 = np.mean(img[805:905, 545:575])
    mean2 = np.mean(img[805:905, 725:755])
    mean3 = np.mean(img[805:905, 905:935])

    return mean1 == 233.0 and mean2 == 233.0 and mean3 == 233.0
    
#testing scoreboard location and threshold values 
def scoreboardLiveTesting(window):
    sct = mss.mss()
    shown = False
    while True:
        img = sct.grab(window)
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if detectScoreboard(img):
            if shown == False:
                print("Scoreboard is up")
                shown = True
        else:
            if shown == True:
                print("Scoreboard is down")
            shown = False

        cv.rectangle(img, (545,805), (575, 905), (255,0,0), 2)
        cv.rectangle(img, (725,805), (755, 905), (255,0,0), 2)
        cv.rectangle(img, (905,805), (935, 905), (255,0,0), 2)

        cv.rectangle(img, (545,855), (575, 905), (0,255,0), 2)
        cv.rectangle(img, (725,855), (755, 905), (0,255,0), 2)
        cv.rectangle(img, (905,855), (935, 905), (0,255,0), 2)

        cv.imshow("Focused", img)
            
        key = cv.waitKey(10)
        if(key == 113):
            break
        elif key == 32:
            mean1 = np.mean(img[805:905, 545:575])
            mean2 = np.mean(img[805:905, 725:755])
            mean3 = np.mean(img[805:905, 905:935])
            print(f"{mean1} {mean2} {mean3}")
        elif key != -1:
            print("Saved image")
    print("DONE")

def encodeNumber(num):
    ret = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if num >= 50 and num <= 57:
        #numerical input
        ret[num - 50] = 1
    elif num == 48:
        # 10
        ret[8] = 1
    elif num == 106:
        # J
        ret[9] = 1
    elif num == 113:
        # Q
        ret[10] = 1
    elif num == 107:
        # K
        ret[11] = 1
    elif num == 97:
        ret[12] = 1
    return ret

def interpreteNumberEncoding(encoding):
    idx = np.argmax(encoding)
    if idx <= 8:
        return f"{idx + 2}"
    elif idx == 9:
        return "Jack"
    elif idx == 10:
        return "Queen"
    elif idx == 11:
        return "King"
    elif idx == 12:
        return "Ace"
    else:
        print("Problem with encoding")
        print(encoding)
    
    
def getDataStatsNumbers():
    fileToY = "LabeledNumbers/TrainingDataY.npy"
    #fileToY = "PredictedNumbers/Y.npy"
    if( os.path.isfile(fileToY)):
        print("File exists")
        trainingDataY = np.load(fileToY)
    else:
        print("File doesn't exist")
        return
    
    count = np.sum(trainingDataY, axis=0)
    print(count)

#manually classify the number value of the saved cards
def classifyNumbers():
    loadXPath = "LabeledSuits/trainingDataX.npy"
    if(os.path.isfile(loadXPath)):
        X = list(np.load(loadXPath))
        print(f"File exists with {len(X)} pictures")
    else:
        print("File doesn't exist")
        return

    loadYPath = "LabeledNumbers/TrainingDataY.npy"
    if(os.path.isfile(loadYPath)):
        Y = list(np.load(loadYPath))
        print(f"File exists with {len(Y)} pictures")
    else:
        print("File doesn't exist")
        return
    
    start = len(Y)
    for i in range(start,len(X)):
        print(i)
        temp = X[i]
        temp = cv.resize(temp, (0,0), fx = 4, fy = 4)
        cv.imshow("card", temp)
        key = cv.waitKey(0)
        if key == 32:
            break
        Y.append(encodeNumber(key))

    print(f"Saving {len(Y)} outputs")
    np.save(loadYPath, Y)

def trainModelNumber():
    Xpath = "LabeledSuits/trainingDataX.npy"
    Ypath = "LabeledNumbers/TrainingDataY.npy"
    if(os.path.isfile(Xpath) and os.path.isfile(Ypath)):
        print("File exists")
        trainingDataX = np.load(Xpath)
        print(f"length {len(trainingDataX)}")
        trainingDataY = np.load(Ypath)
        print(f"length {len(trainingDataY)}")
    else:
        print("File doesn't exist")
        return
    
    #trainingDataX = trainingDataX / 255.0
    trainingDataX = trainingDataX.reshape((len(trainingDataX), 35,25,1))

    modelName = "numberClassifierV2.keras"
    if(os.path.isfile(modelName)):
        print("loading model")
        model = loadModel(modelName)
    else:
        print("creating new model")
        model = createModelNumber()
    
    
    model.fit(trainingDataX, trainingDataY, epochs = 5, batch_size = 16)
    model.save(modelName)
    print("DONE")

#test the model on live game
def liveTestingNumbers(window):
    sct = mss.mss()
    loaded = True
    keepingCount = False
    smallCount = 0
    modelName = "numberClassifierV1.keras"
    if(os.path.isfile(modelName)):
        print("loading model")
        model = loadModel(modelName)
    else:
        print("Model not found")
        return
    
    fileToX = "PredictedNumbers/X.npy"
    fileToY = "PredictedNumbers/Y.npy"
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        imgCnvs = list(np.load(fileToX))
        print(f"length {len(imgCnvs)}")
        predictions = list(np.load(fileToY))
        print(f"length {len(predictions)}")
    else:
        print("File doesn't exist")
        imgCnvs = []
        predictions = []

    testResult = np.zeros((50, 50))
    tricks = 0
    while(True):
        img = sct.grab(window)
        img = np.array(img)
        imgCopy = img.copy()
        arr = parseImage(imgCopy)
        [imgTop, imgBot, imgLeft, imgRight] = arr
        count = 0
        for i in arr:
            temp = np.mean(i)
            if temp > 95:
                count += 1
            
        if count == 4 and loaded:
            loaded = False
            keepingCount = True
        elif count <= 2:
            loaded = True

        if keepingCount == True:
            smallCount += 1

        if smallCount > 10:
            tricks += 1
            keepingCount = False
            smallCount = 0
            #testResult = comb

            imgTopCnv = convertImg(imgTop)
            imgBotCnv = convertImg(imgBot)
            imgLeftCnv = convertImg(imgLeft)
            imgRightCnv = convertImg(imgRight)
            imgCnvs.append(imgTopCnv)
            imgCnvs.append(imgBotCnv)
            imgCnvs.append(imgLeftCnv)
            imgCnvs.append(imgRightCnv)
            
            preTop = model.predict(imgTopCnv)
            preBot = model.predict(imgBotCnv)
            preLeft = model.predict(imgLeftCnv)
            preRight = model.predict(imgRightCnv)

            #topText = interpreteNumberEncoding(preTop[0])
            #botText = interpreteNumberEncoding(preBot[0])
            #leftText = interpreteNumberEncoding(preLeft[0])
            #rightText = interpreteNumberEncoding(preRight[0])

            #cv.putText(mask, topText, (30,90),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            #cv.putText(mask, botText, (130,90),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            #cv.putText(mask, leftText, (30,250),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            #cv.putText(mask, rightText, (130,250),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            #comb = np.concatenate((result, mask))

            #testResult = comb

            predictions.append(preTop[0])
            predictions.append(preBot[0])
            predictions.append(preLeft[0])
            predictions.append(preRight[0])

        #resultVert = np.concatenate((imgTop, imgBot), axis = 1)
        #resultHor = np.concatenate((imgLeft, imgRight), axis = 1)
        #result = np.concatenate((resultVert, resultHor), axis = 0)
        #mask = np.zeros_like(result)
        
        #comb = np.concatenate((result, mask))
        mask = np.zeros_like(testResult)
        cv.putText(mask, f"{tricks}", (10,40),cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv.imshow("Focused", mask)
            
            
        key = cv.waitKey(10)
        if(key == 113):
            break
    
    np.save(fileToX, imgCnvs)
    np.save(fileToY, predictions)
    print(f"final length {len(imgCnvs)}")    
    print(f"final length {len(predictions)}")

#go through the predictions from the model 
def goThroughPredictionNumbers():
    fileToX = "PredictedNumbers/X.npy"
    fileToY = "PredictedNumbers/Y.npy"
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        imgs = np.load(fileToX)
        print(f"length {len(imgs)}")
        print(imgs.shape)
        predictions = np.load(fileToY)
        print(f"length {len(predictions)}")
        print(predictions.shape)
    else:
        print("File doesn't exist")
        return
    
    for i in range(len(imgs)):
        temp = imgs[i]
        temp = temp.reshape((35,25))
        temp = cv.resize(temp, (0,0), fx = 4, fy = 4)
        add = np.zeros_like(temp)
        add = cv.putText(add, f"{interpreteNumberEncoding(predictions[i])}", (0,70), cv.FONT_HERSHEY_COMPLEX, 0.75, (255))
        fin = np.concatenate((temp,add))
        cv.imshow("IMG", fin)
        key = cv.waitKey(0)
        if key == 113:
            break
        elif key == 0:
            print(f"problem on {i}")

#correct any incorrect classifications
def deleteMistake(fileToX, fileToY):
    #fileToX = "PredictedNumbers/X.npy"
    #fileToY = "PredictedNumbers/Y.npy"
    X = list(np.load(fileToX))
    print(f"length {len(X)}")
    Y = list(np.load(fileToY))
    print(f"length {len(Y)}")
    #cv.imshow("test", X[975])
    #cv.waitKey(0)
    #print(Y[975])
    #return 

    X = X[:975]
    Y = Y[:975]
    print(f"length {len(X)}")
    print(f"length {len(Y)}")

    np.save(fileToX, X)
    np.save(fileToY, Y)

def predictData():
    fromXpath = "PredictedNumbers/X.npy"
    toXpath = "PredictedSuits/Imgs.npy"
    toYpath = "PredictedSuits/predictions.npy"

    fromXdata = np.load(fromXpath)
    toXdata = list(np.load(toXpath))
    toYdata = list(np.load(toYpath))
    
    modelName = "suitClassifierv2.keras"
    model = loadModel(modelName)
    
    for i in range(len(fromXdata)):
        temp = fromXdata[i]
        toXdata.append(temp)
        toYdata.append(model.predict(temp)[0])

    np.save(toXpath, toXdata)
    np.save(toYpath, toYdata)

#manually go over predictions from model and correctly classify them 
def classifyNumbersFromPredictions():
    predictionPath = "PredictedNumbers/Y.npy"
    fileX = "LabeledSuits/trainingDataX.npy"
    fileToY = "LabeledNumbers/TrainingDataY.npy"
    
    trainingDataY = []

    if(os.path.isfile(predictionPath)):
        print("Predictions exist")
        predictions = np.load(predictionPath)
        print(f"model predictions shape {predictions.shape}")
    
    if(os.path.isfile(fileX) and os.path.isfile(fileToY)):
        print("File exists")
        X = np.load(fileX)
        trainingDataY = list(np.load(fileToY))
        print(f"Total {len(X)} imgs with {len(trainingDataY)} classifications already done")
        print(f"{len(X) - len(trainingDataY)} classifications need to be done")
    else:
        print("File doesn't exist")
        return
        
    
    
    beg = 0
    end = len(X)

    print(f"starting on {beg}")
    print(f"total to do {end}")
    toDo = end - beg
    i = 0
    right = 0
    wrong = 0
    while beg < end:
        img = X[beg]
        img = img.reshape(35,25)
        big = cv.resize(img, (0,0), fx = 4, fy = 4)
        p = predictions[i]

        pre = np.zeros_like(big)
        pre = cv.putText(pre, interpreteNumberEncoding(p), (20,70), cv.FONT_HERSHEY_COMPLEX, 2, (255))
        show = np.concatenate((big, pre))
        cv.imshow("IMG",show)

        key = cv.waitKey(0)
        
        encoding = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        if key == 32:
            break
        elif key == 3:
            idx = np.argmax(p)
            encoding[idx] = 1
            right += 1
            print(f"Correct {beg}")
        else:
            encoding = encodeNumber(key)
            wrong += 1
            print(f"Incorrect {beg}")
        
        encoding = np.array(encoding)
        
        
        #trainingDataX.append(cv.resize(img, (0,0), fx = .25, fy = .25))
        trainingDataY.append(encoding)
        beg += 1
        i += 1
    print(f"{right} right and {wrong} wrong")
    print(f"SAVING {len(trainingDataY)}")
    

    #np.save(fileToY, trainingDataY)    

#check the classification of a specific image
def checkNumberClassifications(idx):
    fileToY = "LabeledNumbers/TrainingDataY.npy"
    trainingDataY = []
    fileToX= "LabeledSuits/trainingDataX.npy"
    trainingDataX = []
    if(os.path.isfile(fileToY) and os.path.isfile(fileToX)):
        print("File exists")
        trainingDataX = list(np.load(fileToX))
        print(f"length {len(trainingDataX)}")
        trainingDataY = list(np.load(fileToY))
        print(f"length {len(trainingDataY)}")
    else:
        print("File doesn't exist")
        return
    
    for i in range(len(trainingDataX)):
        encoding = trainingDataY[i]
        if(encoding[idx] != 1):
            continue
        temp = trainingDataX[i]
        
        
        temp = cv.resize(temp, (0,0), fx = 4, fy = 4)
        add = np.zeros_like(temp)
        add = cv.putText(add, f"{i}", (20,70), cv.FONT_HERSHEY_COMPLEX, 2, (255))
        fin = np.concatenate((temp,add))
        cv.imshow("IMG", fin)
        key = cv.waitKey(0)
        if key == 113:
            break
        elif key == 0:
            print(f"problem on {i}")

#go through training data 20 at a time
def goThroughTrainingData():
    fileToX= "LabeledSuits/trainingDataX.npy"
    if(os.path.isfile(fileToX)):
        print("File exists")
        X = np.load(fileToX)
        print(f"length {len(X)}")
        print(X.shape)
    else:
        print("File doesn't exist")
        return
    
    
    
    for i in range(0,len(X) - 20,20):
        print(i)
        show = np.zeros((70,50))
        for j in range(10):
            temp = X[i + j]
            temp = temp.reshape(35,25)
            temp = cv.resize(temp, (0,0), fx= 2, fy = 2)
            show = np.concatenate((show, temp), axis = 1)
        show1 = np.zeros((70,50))
        for j in range(10):
            temp = X[i + j + 10]
            temp = temp.reshape(35,25)
            temp = cv.resize(temp, (0,0), fx= 2, fy = 2)
            show1 = np.concatenate((show1, temp), axis = 1)
        final = np.concatenate((show, show1))
        cv.imshow("Test", final)
        key = cv.waitKey(0)
        if(key == 32):
            break
        
def showAreasOfInterest():
    window = {"top":20, "left":1050, "width":740, "height":980}
    sct = mss.mss()
    height = 140
    width = 100
    
    angle1 = -0.7083333333333334
    l1 = [375, 925]
    l2 = [width * math.cos(angle1) + l1[0], width * math.sin(angle1) + l1[1]]
    l3 = [width * math.cos(angle1) + height * math.sin(-angle1) + l1[0], width * math.sin(angle1) + height * math.cos(-angle1) + l1[1]]
    l4 = [height * math.sin(-angle1) + l1[0], height * math.cos(-angle1) + l1[1]]
    l2[0] = math.ceil(l2[0])
    l2[1] = math.ceil(l2[1])
    l3[0] = math.ceil(l3[0])
    l3[1] = math.ceil(l3[1])
    l4[0] = math.ceil(l4[0])
    l4[1] = math.ceil(l4[1])

    angle2 = 1.0833333333333333
    r1 = [980, 705]
    r2 = [width * math.cos(angle2) + r1[0], width * math.sin(angle2) + r1[1]]
    r3 = [width * math.cos(angle2) + height * math.sin(-angle2) + r1[0], width * math.sin(angle2) + height * math.cos(-angle2) + r1[1]]
    r4 = [height * math.sin(-angle2) + r1[0], height * math.cos(-angle2) + r1[1]]
    r2[0] = math.ceil(r2[0])
    r2[1] = math.ceil(r2[1])
    r3[0] = math.ceil(r3[0])
    r3[1] = math.ceil(r3[1])
    r4[0] = math.ceil(r4[0])
    r4[1] = math.ceil(r4[1])
    

    while True:
        img = sct.grab(window)
        img = np.array(img)
        imgCopy = img.copy()
        cv.rectangle(imgCopy, (565, 635), (565 + width, 635 + height),(255,0,0), 2)
        cv.rectangle(imgCopy, (795, 1155), (795 + width, 1155 + height), (255,0,0), 2)
        
        cv.line(imgCopy, l1, l2, (255,0,0), 2)
        cv.line(imgCopy, l2, l3, (255,0,0), 2)
        cv.line(imgCopy, l3, l4, (255,0,0), 2)
        cv.line(imgCopy, l4, l1, (255,0,0), 2)

        cv.line(imgCopy, r1, r2, (255,0,0), 2)
        cv.line(imgCopy, r2, r3, (255,0,0), 2)
        cv.line(imgCopy, r3, r4, (255,0,0), 2)
        cv.line(imgCopy, r4, r1, (255,0,0), 2)


        cv.imshow("Focused", imgCopy)

        key = cv.waitKey(10)
        if(key == 113):
            break