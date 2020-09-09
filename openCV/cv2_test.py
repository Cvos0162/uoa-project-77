#Grouping of contours for classification:
#to close opened windows, focus on any window and press ESC (or manually 'x' out of each one)

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path

class Box:
    def __init__(self, startX, startY, width, height):
        self.x = startX
        self.y = startY
        self.width = width
        self.height = height

#CONVERTING PDF TO IMAGE
#pages = convert_from_path('NZBC-G4#3.4.pdf', 500)

#SAVING FIRST PAGE
#pages[20].save('page20.jpg', 'JPEG')

#threshold value (image binary thresholding)
threshold = 200

#distance thresholds (contour association)
cDistance = 55

#read page, convert to grayscale and initial threshold
page = cv2.imread('page18.jpg')
page_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
ret, page_thresh = cv2.threshold(page_gray, threshold, 255, cv2.THRESH_BINARY)

pageHeight,pageWidth,_ = page.shape

#threshold: prevent contour around page border being considered
contourOversize = (pageHeight*pageWidth)/2

page_disp = page.copy()
contour_disp = page.copy()
grouped_disp = page.copy()
thresh_disp = page_thresh.copy()

#find contours
contours,_ = cv2.findContours(page_thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)

#if you want to simply draw all contours
cv2.drawContours(contour_disp, contours, -1, (0,0,255), 1)

boxes1 = []

while len(contours) > 1:
    xStart = pageWidth-1
    xEnd = 0
    yStart = pageHeight-1
    yEnd = 0

    for j in range(15):
        for i in range(len(contours)):
            contour = contours[i]
            (x,y,w,h) = cv2.boundingRect(contour)

            #ignore large contours (page border)
            if cv2.contourArea(contour) > contourOversize:
                continue

            if xEnd == 0 or yEnd == 0:
                xStart = x
                yStart = y
                xEnd = x+w
                yEnd = y+h
                
            if (x < xEnd+cDistance and x > xStart-cDistance and y < yEnd+cDistance and y > yStart-cDistance) or \
               (x+w < xEnd+cDistance and x+w > xStart-cDistance and y+h < yEnd+cDistance and y+h > yStart-cDistance):
                if x < xStart:
                    xStart = x
                if x+w > xEnd:
                    xEnd = x+w
                if y < yStart:
                    yStart = y
                if y+h > yEnd:
                    yEnd = y+h

    wGroup = xEnd - xStart
    hGroup = yEnd - yStart

    #create new boundingBox
    cv2.rectangle(grouped_disp, (xStart,yStart),(xEnd,yEnd),(0,0,255),2)
    width = xEnd - xStart
    height = yEnd - yStart
    newBox = Box(xStart, yStart, width, height)
    boxes1.append(newBox)
    
    #remove contours inside element box from list
    for i in range(len(contours)-1,-1,-1):
        contour = contours[i]
        (x,y,w,h) = cv2.boundingRect(contour)

        if (x < xEnd and x >= xStart) and (y < yEnd and y >= yStart):
            del contours[i]

print("NO. IDENTIFIED ELEMENTS: "+str(len(boxes1)))

#extraction of templates from existing pages
#box = boxes1[18]
#createTemp = page[box.y:box.y+box.height, box.x:box.x+box.width]
#saveTemp = createTemp[0:60, 0:500]
#cv2.imshow('element', createTemp)
#cv2.imshow('template', saveTemp)
#cv2.imwrite('subsection5.jpg', saveTemp)

#load in templates
commentT = []
listT = []
paragraphT = []
subsectionT = []
topicT = []
for i in range(5):
    commentString = 'Element-Templates/comment'+str(i+1)+'.jpg'
    listString = 'Element-Templates/list'+str(i+1)+'.jpg'
    paragraphString = 'Element-Templates/paragraph'+str(i+1)+'.jpg'
    subsectionString = 'Element-Templates/subsection'+str(i+1)+'.jpg'
    topicString = 'Element-Templates/topic'+str(i+1)+'.jpg'
    commentT.append(cv2.imread(commentString, cv2.IMREAD_GRAYSCALE))
    listT.append(cv2.imread(listString, cv2.IMREAD_GRAYSCALE))
    paragraphT.append(cv2.imread(paragraphString, cv2.IMREAD_GRAYSCALE))
    subsectionT.append(cv2.imread(subsectionString, cv2.IMREAD_GRAYSCALE))
    topicT.append(cv2.imread(topicString, cv2.IMREAD_GRAYSCALE))

#calculate similarity between each element and various templates to find
#closest match
for box in boxes1:
    element = page[box.y:box.y+box.height, box.x:box.x+box.width]
    element = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)

    commentCheck = 0
    listCheck = 0
    paragraphCheck = 0
    subsectionCheck = 0
    topicCheck = 0
    highSim = 0
    high = ''

    #COMMENT SIMILARITY
    for i in range(len(commentT)):
        commentImg = commentT[i]

        height,width = element.shape
        cHeight,cWidth = commentImg.shape

        if height < cHeight:
            height = cHeight
        if width < cWidth:
            width = cWidth

        bufferE = np.ones([height, width], np.uint8)
        bufferE[0:len(element), 0:len(element[0])] = element

        result = cv2.matchTemplate(bufferE, commentImg, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        commentCheck += max_val

    commentCheck = commentCheck/len(commentT)
    highSim = commentCheck
    high = 'comment'

    #LIST SIMILARITY
    for i in range(len(listT)):
        listImg = listT[i]
        
        height,width = element.shape
        cHeight,cWidth = listImg.shape

        if height < cHeight:
            height = cHeight
        if width < cWidth:
            width = cWidth

        bufferE = np.ones([height, width], np.uint8)
        bufferE[0:len(element), 0:len(element[0])] = element

        result = cv2.matchTemplate(bufferE, listImg, cv2.TM_CCOEFF_NORMED)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        listCheck += max_val

    listCheck = listCheck/len(listT)
    if (listCheck > highSim):
        highSim = listCheck
        high = 'list'

    #PARAGRAPH SIMILARITY
    for i in range(len(paragraphT)):
        paragraphImg = paragraphT[i]

        height,width = element.shape
        cHeight,cWidth = paragraphImg.shape

        if height < cHeight:
            height = cHeight
        if width < cWidth:
            width = cWidth

        bufferE = np.ones([height, width], np.uint8)
        bufferE[0:len(element), 0:len(element[0])] = element

        result = cv2.matchTemplate(bufferE, paragraphImg, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        paragraphCheck += max_val

    paragraphCheck = paragraphCheck/len(paragraphT)
    if (paragraphCheck > highSim):
        highSim = paragraphCheck
        high = 'paragraph'

    #SUBSECTION SIMILARITY
    for i in range(len(subsectionT)):
        subsectionImg = subsectionT[i]

        height,width = element.shape
        cHeight,cWidth = subsectionImg.shape

        if height < cHeight:
            height = cHeight
        if width < cWidth:
            width = cWidth

        #bypass if element is particularly large as it is unlikely to be
        #a subsection title
        if height > cHeight*3:
            continue

        bufferE = np.ones([height, width], np.uint8)
        bufferE[0:len(element), 0:len(element[0])] = element

        result = cv2.matchTemplate(bufferE, subsectionImg, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        subsectionCheck += max_val

    subsectionCheck = subsectionCheck/len(subsectionT)
    if (subsectionCheck > highSim):
        highSim = subsectionCheck
        high = 'sub'

    #TOPIC SIMILARITY
    for i in range(len(topicT)):
        topicImg = topicT[i]

        height,width = element.shape
        cHeight,cWidth = topicImg.shape

        if height < cHeight:
            height = cHeight
        if width < cWidth:
            width = cWidth

        bufferE = np.ones([height, width], np.uint8)
        bufferE[0:len(element), 0:len(element[0])] = element

        result = cv2.matchTemplate(bufferE, topicImg, cv2.TM_CCOEFF_NORMED)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        topicCheck += max_val

    topicCheck = topicCheck/len(topicT)
    if (topicCheck > highSim):
        highSim = topicCheck
        high = 'topic'

    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(grouped_disp, high, (box.x,box.y), font,2,(0,0,255),2,cv2.LINE_AA)

#determining order of elements on the page
lowX = 5000
highX = 0
highY = 0
for i in range(len(boxes1)):
    if boxes1[i].x < lowX:
        lowX = boxes1[i].x
    if boxes1[i].x > highX:
        highX = boxes1[i].x

for i in range(len(boxes1)):
    if x < lowX+500:
        if boxes1[i].y > highY:
            highY = boxes1[i].y

for j in range(len(boxes1)):
    lowIndex = j
    lowY = 6000
    xList = []
    
    for i in range(j, len(boxes1)):
        x = boxes1[i].x
        y = boxes1[i].y

        if x < lowX+500:
            if y < lowY:
                lowY = y
                lowIndex = i

        if i == len(boxes1)-1:
            if lowY == highY:
                lowX = highX
        
    temp = boxes1[j]
    boxes1[j] = boxes1[lowIndex]
    boxes1[lowIndex] = temp

#print order numbers next to element boxes
for i in range(len(boxes1)):
    x = boxes1[i].x
    y = boxes1[i].y
    elementNum = str(i)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grouped_disp, elementNum, (x,y), font,2,(0,0,255),2,cv2.LINE_AA)

#image is 4134x5847 scale down
for i in range(3):
    #thresh_disp = cv2.pyrDown(thresh_disp)
    #contour_disp = cv2.pyrDown(contour_disp)
    grouped_disp = cv2.pyrDown(grouped_disp)

#cv2.imshow('page', page_disp)   
#cv2.imshow('thresholded', thresh_disp)
#cv2.imshow('contoured', contour_disp)
cv2.imshow('grouped', grouped_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()



