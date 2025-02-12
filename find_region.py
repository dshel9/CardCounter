import math
import cv2 as cv

def up(points, delta):
    for point in points:
        point[1] -= delta

def down(points, delta):
    for point in points:
        point[1] += delta

def left(points, delta):
    for point in points:
        point[0] -= delta

def right(points, delta):
    for point in points:
        point[0] += delta

def shorter(points, delta):
    points[2][1] -= delta
    points[3][1] -= delta

def taller(points, delta):
    points[2][1] += delta
    points[3][1] += delta

def thinner(points, delta):
    points[1][0] -= delta
    points[2][0] -= delta

def wider(points, delta):
    points[1][0] += delta
    points[2][0] += delta

def rotate(points, angle, h, w):
    x = points[0][0]
    y = points[0][1]

    points[1][0] = w * math.cos(angle) + x
    points[1][1] = w * math.sin(angle) + y

    points[2][0] =  w * math.cos(angle) + h * math.sin(angle * -1) + x
    points[2][1] = w * math.sin(angle) + h * math.cos(angle * -1) + y 

    points[3][0] = h * math.sin(angle * -1) + x
    points[3][1] = h * math.cos(angle * -1) + y 

    


def findScoreBoard(color):
    p1 = [25, 25]
    p2 = [125, 25]
    p3 = [125, 125]
    p4 = [25, 125]
    points = [p1, p2, p3, p4]
    delta = 10
    i = 6
    while True:
        img = cv.imread(f"TestIMG/ImageTest{i}.png", cv.IMREAD_UNCHANGED)
        imgCopy = img.copy()
        cv.line(imgCopy, p1, p2, color, 2)
        cv.line(imgCopy, p2, p3, color, 2)
        cv.line(imgCopy, p3, p4, color, 2)
        cv.line(imgCopy, p4, p1, color, 2)
        cv.imshow("Current box", imgCopy)
        key = cv.waitKey(1)
        if(key == 113):
            break
        elif key == 0:
            #up
            up(points, delta)
            
        elif key == 1:
            #down
            down(points, delta)
            
        elif key == 2:
            #left
            left(points, delta)
        elif key == 3:
            #right
            right(points, delta)
        elif key == 119:
            #shorter
            shorter(points, delta)
        elif key == 115:
            #taller
            taller(points, delta)
        elif key == 97:
            #skinner
            thinner(points, delta)
        elif key == 100:
            #wider
            wider(points, delta)
        elif key == 44:
            #prev photo
            i -= 1
            i = max(2, i)
        elif key == 46:
            #next photo
            i += 1
            i = min(i, 5)
        elif key == 32:
            print(f"p1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}")
            print(f"Top left {p1}, with width of {p2[0] - p1[0]}, and height of {p3[1] - p2[1]}")
        elif key != -1:
            print(key)
    print(f"p1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}")
    print(f"Top left {p1}, with width of {p2[0] - p1[0]}, and height of {p3[1] - p2[1]}")

def findInitialRegion(img, color):
    p1 = [25, 25]
    p2 = [125, 25]
    p3 = [125, 125]
    p4 = [25, 125]
    points = [p1, p2, p3, p4]
    delta = 10
    while True:
        imgCopy = img.copy()
        cv.line(imgCopy, p1, p2, color, 2)
        cv.line(imgCopy, p2, p3, color, 2)
        cv.line(imgCopy, p3, p4, color, 2)
        cv.line(imgCopy, p4, p1, color, 2)
        cv.imshow("Current box", imgCopy)
        key = cv.waitKey(1)
        if(key == 113):
            break
        elif key == 0:
            #up
            up(points, delta)
        elif key == 1:
            #down
            down(points, delta)
        elif key == 2:
            #left
            left(points, delta)
        elif key == 3:
            #right
            right(points, delta)

        elif key == 119:
            #shorter
            shorter(points, delta)
        elif key == 115:
            #taller
            taller(points, delta)

        elif key == 97:
            #skinner
            thinner(points, delta)
        elif key == 100:
            #wider
            wider(points, delta)

        elif key == 122:
            #decrease delta
            delta -= 5
        elif key == 120:
            #increase delta
            delta += 5
        elif key != -1:
            print(key)
    print(f"p1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}")
    print(f"Top left {p1}, with width of {p2[0] - p1[0]}, and height of {p3[1] - p2[1]}")


def findBottomRegion(width, height, img, color):
    p1 = [25, 25]
    p2 = [25 + width, 25]
    p3 = [25 + width, 25 + height]
    p4 = [25, 25 + height]
    points = [p1, p2, p3, p4]
    delta = 10
    while True:
        imgCopy = img.copy()
        
        cv.line(imgCopy, p1, p2, color, 2)
        cv.line(imgCopy, p2, p3, color, 2)
        cv.line(imgCopy, p3, p4, color, 2)
        cv.line(imgCopy, p4, p1, color, 2)
        cv.imshow("Current box", imgCopy)

        key = cv.waitKey(10)
        if(key == 113):
            break
        elif key == 0:
            #up
            up(points, delta)
        elif key == 1:
            #down
            down(points, delta)
        elif key == 2:
            #left
            left(points, delta)
        elif key == 3:
            #right
            right(points, delta)

        elif key == 122:
            #decrease delta
            delta -= 5
        elif key == 120:
            #increase delta
            delta += 5
        elif key != -1:
            print(key)
    print(f"p1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}")
    print(f"Top left {p1}, with width of {p2[0] - p1[0]}, and height of {p3[1] - p2[1]}")

def findSideCards(width, height, img, color):
    angle = 0
    p1 = [25, 25]
    p2 = [25 + width, 25]
    p3 = [25 + width, 25 + height]
    p4 = [25, 25 + height]
    points = [p1, p2, p3, p4]
    delta1 = 10
    delta2 = 1/6
    while True:
        imgCopy = img.copy()
        #cv.rectangle(imgCopy, (x,y), (x + w, y + h), color, 2)
        #print(f"{type(p2)} {type(p2[0])} {type(p2[1])} {type(round(p2[1], 0))}")
        
        cv.line(imgCopy, p1, (math.ceil(p2[0]), math.ceil(p2[1])), color, 2)
        cv.line(imgCopy, (math.ceil(p2[0]), math.ceil(p2[1])), (math.ceil(p3[0]), math.ceil(p3[1])), color, 2)
        cv.line(imgCopy, (math.ceil(p3[0]), math.ceil(p3[1])), (math.ceil(p4[0]), math.ceil(p4[1])), color, 2)
        cv.line(imgCopy, (math.ceil(p4[0]), math.ceil(p4[1])), p1, color, 2)
        cv.imshow("Current box", imgCopy)
        key = cv.waitKey(1)
        if(key == 113):
            break
        elif key == 0:
            #up
            up(points, delta1)
            
        elif key == 1:
            #down
            down(points, delta1)
            
        elif key == 2:
            #left
            left(points, delta1)
        elif key == 3:
            #right
            right(points, delta1)

        elif key == 44:
            #rotate left
            angle -= delta2
            rotate(points, angle, height, width)
        elif key == 46:
            #rotate left
            angle += delta2
            rotate(points, angle, height, width)

        elif key == 122:
            #decrease delta
            delta1 -= 5
            delta2 /= 2
        elif key == 120:
            #increase delta
            delta1 += 5
            delta2 *= 2
        elif key != -1:
            print(key)

    print(p1)
    print(angle)

def outlineCard(img, color):
    p1 = [25,25]
    p2 = [125, 25]
    delta = 10
    while True:
        imgCopy = img.copy()
        cv.line(imgCopy, p1, p2, color, 4)
        cv.imshow("Picture",imgCopy)
        key = cv.waitKey(25)
        if(key == 113):
            break
        elif key == 0:
            #up
            p2[1] -= delta
            
        elif key == 1:
            #down
            p2[1] += delta
            
        elif key == 2:
            #left
            p2[0] -= delta
        elif key == 3:
            #right
            p2[0] += delta
        elif key == 119:
            #up
            p1[1] -= delta
        elif key == 115:
            #down
            p1[1] += delta
        elif key == 97:
            #left
            p1[0] -= delta
        elif key == 100:
            #right
            p1[0] += delta
        elif key == 122:
            #decrease delta
            delta -= 5
        elif key == 120:
            #increase delta
            delta += 5
    print(p1)
    print(p2)
