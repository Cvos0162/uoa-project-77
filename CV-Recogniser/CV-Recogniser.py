#Instructions for running:
#1: ensure appropriate libraries/source files are all installed including numpy, openCV, 
#   pdf2image (which may involve poppler/others), tesseract.
#2: edit lines 37 to desired document; ensuring path is correct
#3: output.xml should be produced in same directory as this python script.


import xml.etree.ElementTree as xml
import xml.dom.minidom as minidom

import math
import cv2
import numpy as np
from pdf2image import convert_from_path

import pytesseract

class Box:
    def __init__(self, startX, startY, width, height):
        self.x = startX
        self.y = startY
        self.width = width
        self.height = height
        self.type = "unclassified"
        self.content = ""

#Below code for getting images of pages for processing (only needs to be done once,
#takes some time)
#pages = convert_from_path('NZBC-G4#3.4.pdf', 500) #CONVERTING PDF TO IMAGE
#pages[14].save('G4-page13.jpg', 'JPEG') #SAVING PAGE

#Threshold parameters; these would need to be adjusted depending on the document
threshold = 200 #threshold value (image binary thresholding)
cDistance = 55 #distance threshold (contour association)

#read page, convert to grayscale and initial threshold
page = cv2.imread('Test-Pages\G4-page14.jpg')
page_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
ret, page_thresh = cv2.threshold(page_gray, threshold, 255, cv2.THRESH_BINARY)

pageHeight,pageWidth,_ = page.shape

#contour size threshold: i.e. prevent contour around page border being considered
contourOversize = (pageHeight*pageWidth)/2

page_disp = page.copy()
contour_disp = page.copy()
grouped_disp = page.copy()
thresh_disp = page_thresh.copy()

#find contours
contours,_ = cv2.findContours(page_thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)

#if you want to simply draw all contours
cv2.drawContours(contour_disp, contours, -1, (0,0,255), 2)

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

    #create new boundingBox for element then add element to 'boxes1' list
    #cv2.rectangle(grouped_disp, (xStart,yStart),(xEnd,yEnd),(0,0,255),2) (LINE FOR DEBUG)
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


#DETERMINING ELEMENT ORDER
lowX = 5000
lowY = 5000
highX = 0
highY = 0
#lowest and highest x value of elements
for i in range(len(boxes1)):
    if boxes1[i].x < lowX:
        lowX = boxes1[i].x
    if boxes1[i].x > highX:
        highX = boxes1[i].x
    if boxes1[i].y < lowY:
        lowY = boxes1[i].y

#highest y value of the left column (500 px threshold for associating with
#left column)
for i in range(len(boxes1)):
    if x < lowX+750:
        if boxes1[i].y > highY:
            highY = boxes1[i].y

#find appropriate element for 1st box then second etc. until end of first
#column is reached (highY) then do the same for the second column
for j in range(len(boxes1)):
    lowIndex = j
    currentY = 6000
    xList = []
    
    for i in range(j, len(boxes1)):
        x = boxes1[i].x
        y = boxes1[i].y

        if x < lowX+750:
            if y < currentY:
                currentY = y
                lowIndex = i

        if i == len(boxes1)-1:
            if currentY == highY:
                lowX = highX
        
    temp = boxes1[j]
    boxes1[j] = boxes1[lowIndex]
    boxes1[lowIndex] = temp
    

#PERFORM TEMPLATE MATCHING; for classifying elements
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

    if box.y > lowY-25 and box.y < lowY+25:
        print("HEADER: "+str(high)+", "+str(highSim))
    elif box.y > highY-25 and box.y < highY+25:
        print("FOOTER: "+str(high)+", "+str(highSim))
    else:
        box.type = high


#grouping of subsection + topic elements considered to be on the same line in the same
#column (specific to the NZ building code documents)
#extract text check for any paragraph/list elements which do NOT contain numbers
#i.e. 1.4.1, 2.3.1, a), c) and group with previous element for para/list
#don't perform these checks for first element (i==0)
i = 0
img = cv2.cvtColor(page, cv2.COLOR_BGR2RGB) #pytesseract works with RGB
while i < len(boxes1):  
    x = boxes1[i].x
    y = boxes1[i].y
    width = boxes1[i].width
    height = boxes1[i].height
    boxType = boxes1[i].type
    prevType = boxes1[i-1].type

    #ignore header/footer and page numbers
    if boxType == "unclassified":
        i += 1
        continue

    #for particularly small elements text may not be picked up to avoid this create buffer
    #around edge of element for OCR (note: the buffer must be smaller than the contouring
    #threshold used to avoid picking up other elements text(20 px here smaller than 55 px
    #used for contouring)
    yStart = 0
    yEnd = 0
    xStart = 0
    xEnd = 0
    if y-20 > 0:
        yStart = y-20
    else:
        yStart = 0
    if (y+20+height) < pageHeight:
        yEnd = y+20+height
    else:
        yEnd = pageHeight
    if x-20 > 0:
        xStart = x-20
    else:
        xStart = 0
    if (x+20+width) < pageWidth:
        xEnd = x+20+width
    else:
        xEnd = pageWidth
    
    elementImg = img[yStart:yEnd, xStart:xEnd]
    elementString = pytesseract.image_to_string(elementImg)
    elementString = elementString.replace('\n', ' ')
    boxes1[i].content = elementString

    #this code could likely be removed with a more extensive set of templates, due to misclassification
    #of paragraphs as lists and lists and topics, this catches this by checking for the list/paragraph
    #indicators
    if len(elementString) >= 4:
        if (elementString[1] == ')' or elementString[2] == ')' or elementString[3] == ')'):
            boxType = "list"
            boxes1[i].type = "list"
    if len(elementString) >= 5:
        if (ord(elementString[0]) > 47 and ord(elementString[0]) < 58) and elementString[1] == '.' and \
            (ord(elementString[2]) > 47 and ord(elementString[2]) < 58) and elementString[3] == '.' and \
            (ord(elementString[4]) > 47 and ord(elementString[4]) < 58):
                boxType = "paragraph"
                boxes1[i].type = "paragraph"
    
    if i > 0:
        group = False

        #check if current and previous type where both sub/topics, reallocate to sub and
        #set flag for grouping
        if boxType == "sub" or boxType == "topic" and \
            prevType == "sub" or prevType == "topic":
                if boxes1[i].y < boxes1[i-1].y+50 and boxes1[i].y > boxes1[i-1].y-50:
                    boxType = "sub"
                    group = True
        elif boxType == "list" or boxType == "paragraph" and \
             prevType == "list" or prevType == "paragraph":
                if boxes1[i].y < boxes1[i-1].y+50 and boxes1[i].y > boxes1[i-1].y-50:
                    group = True
                    
        #if type is paragraph but does not contain number value i.e 1.3.2 then group with previous element
        #if type is list but does not contain list value i.e a), b), i), ii), iii) then group with previous element
        #additional check to ensure sublists are not being grouped with lists (grouping in y therefore x-values
        #must be similar)
        if len(elementString) >= 5 and group == False:
            if boxType == "paragraph":
                if not ((ord(elementString[0]) > 47 and ord(elementString[0]) < 58) and elementString[1] == '.' and \
                (ord(elementString[2]) > 47 and ord(elementString[2]) < 58) and elementString[3] == '.' and \
                (ord(elementString[4]) > 47 and ord(elementString[4]) < 58)):
                    if boxes1[i].x > boxes1[i-1].x-20 and boxes1[i].x < boxes1[i-1].x+20:
                        group = True
                        
            elif boxType == "list":
                if not (elementString[1] == ')' or elementString[2] == ')' or elementString[3] == ')'):
                    if boxes1[i].x > boxes1[i-1].x-20 and boxes1[i].x < boxes1[i-1].x+20:
                        group = True

        if group:
            x = min(boxes1[i].x, boxes1[i-1].x)
            y = min(boxes1[i].y, boxes1[i-1].y)

            #new width must take gap between elements into account
            xGap = 0
            yGap = 0
            if boxes1[i].x < boxes1[i-1].x:
                xGap = boxes1[i-1].x - (boxes1[i].x + boxes1[i].width)
            else:
                 xGap = boxes1[i].x - (boxes1[i-1].x + boxes1[i-1].width)  
            if boxes1[i].y < boxes1[i-1].y:
                yGap = boxes1[i-1].y - (boxes1[i].y + boxes1[i].height)
            else:
                yGap = boxes1[i].y - (boxes1[i-1].y + boxes1[i-1].height)

            if xGap > 0:
                width = boxes1[i].width + xGap + boxes1[i-1].width
            else:
                width = max(boxes1[i].width, boxes1[i-1].width)
            if yGap > 0:
                height = boxes1[i].height + yGap + boxes1[i-1].height
            else:
                height = max(boxes1[i].height, boxes1[i-1].height)
                
            boxType = boxes1[i-1].type

            boxes1[i-1].x = x
            boxes1[i-1].y = y
            boxes1[i-1].width = width
            boxes1[i-1].height = height

            #extract text from combined boxes to get correct element text
            elementImg = img[(y-20):(y+20)+height, (x-20):(x+20)+width]
            elementString = pytesseract.image_to_string(elementImg)
            elementString = elementString.replace('\n', ' ')
            
            boxes1[i-1].content = elementString

            del boxes1[i]
            i -= 1
            
    boxes1[i].type = boxType
    i += 1
    

#run through final boxes list; check subsections for number value (e.g 1.2, 1.3) reclassify as topic for subsections
#that do not have this then produce XML(legalDocML)
root = xml.Element("document")
newParagraph = True
newList = True
indented = False
parentListX = 0
endIndex = 0
startIndex = 0

#default elements incase for example list occurs before paragraph occurs (would be impossible with full
#document)
subsection = xml.SubElement(root, "subsection")
paragraph = xml.SubElement(subsection, "paragraph")
li = xml.SubElement(paragraph, "li")
subsection.set("key", "default")
paragraph.set("key", "default")
li.set("key", "default")

for i in range(len(boxes1)):
    x = boxes1[i].x
    y = boxes1[i].y
    width = boxes1[i].width
    height = boxes1[i].height
    boxType = boxes1[i].type
    elementString = boxes1[i].content

    if len(elementString) < 1:
        boxType = "unclassified"
        boxes1[i].type = "unclassified"

    #ignore header/footer and page numbers
    if boxType == "unclassified":
        i += 1
        continue

    #Running on a laptop this additional code was needed before checking the ords below, 
    #due to tesseract outputting empty strings for smaller elements, additionally random FF 
    #(form-feed characters were present and are removed in the below line)
    #elementString = elementString.replace('\f', '')
    #if len(elementString) > 1:
    #	...(ord code)
    #else:
    #	boxType = "topic"

    #check for initial number for subsection i.e. 1.1, 2.3, if not present reclassify as
    #topic and vice-versa
    if boxType == "sub" and len(elementString) >= 3:
        if not ((ord(elementString[0]) > 47 and ord(elementString[0]) < 58) and elementString[1] == '.' and \
        (ord(elementString[2]) > 47 and ord(elementString[2]) < 58)):
            boxType = "topic"
    elif boxType == "topic" and len(elementString) >= 3:
        if (ord(elementString[0]) > 47 and ord(elementString[0]) < 58) and elementString[1] == '.' and \
        (ord(elementString[2]) > 47 and ord(elementString[2]) < 58):
            boxType = "sub"

    #reclassifying indented list elements as sublist
    if boxType == "list":
        if boxes1[i-1].type == "list":
            if boxes1[i].x > boxes1[i-1].x+50:
                boxType = "sublist"
                parentListX = boxes1[i-1].x
                indented = True
        if indented == True and boxes1[i].x > parentListX+50:
            boxType = "sublist"
        else:
            indented = False    

    boxes1[i].type = boxType

    if len(elementString) < 1:
        boxType = "unclassified"
        boxes1[i].type = "unclassified"

    #ignore header/footer and page numbers
    if boxType == "unclassified":
        i += 1
        continue

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grouped_disp, boxType, (x,y-5), font,2,(0,0,255),2,cv2.LINE_AA)
    cv2.rectangle(grouped_disp, (x,y),(x+width,y+height),(0,0,255),2)

    print("#####"+boxes1[i].type.upper()+"#####")
    print(boxes1[i].content)
    print("["+str(len(boxes1[i].content))+"]")
    print("-----------------------")

    #logic for writing the box content into appropriate XML(legalDocML) format
    if boxType == "sub":
        subsection = xml.SubElement(root, "subsection")
        subsection.set("key", elementString[0:3])

        #finding starting point of title (ignoring dashes/spaces etc.)
        for i in range(len(elementString)):
            if ord(elementString[i]) >= 65:
                startIndex = i
                break
            
        subsection.set("title", elementString[startIndex:len(elementString)])
    elif boxType == "topic":
        topic = xml.SubElement(subsection, "topic")
        topic.text = elementString
    elif boxType == "paragraph":
        paragraph = xml.SubElement(subsection, "paragraph")
        paragraph.set("key", elementString[0:5])
        p = xml.SubElement(paragraph, "p")

        #finding starting point of paragraph text (ignoring dashes/spaces etc.)
        for i in range(len(elementString)):
            if ord(elementString[i]) >= 65:
                startIndex = i
                break
                      
        p.text = elementString[startIndex:len(elementString)]
        newParagraph = True
    elif boxType == "comment":
        if indented == True:
            comment = xml.SubElement(subOl, "commentary")
        else:
            comment = xml.SubElement(paragraph, "commentary")

        #finding ending point of comment title (usually "COMMENT")
        for i in range(len(elementString)):
            if ord(elementString[i]) < 65:
                endIndex = i
                break

        #finding start of comment after title
        for i in range(endIndex, len(elementString)):
            if ord(elementString[i]) < 65:
                startIndex = i
                break
        
        comment.set("title", elementString[0:endIndex])
        comment.text = elementString[startIndex:len(elementString)]
    elif boxType == "list":
        if newParagraph:
            ol = xml.SubElement(paragraph, "ol")
            newParagraph = False

        li = xml.SubElement(ol, "li")
        li.set("key", paragraph.get("key")+"."+elementString[0])

        newList = True

        #finding list content after list 'title' (such as a), b), c) etc.)
        for i in range(2, len(elementString)):
            if ord(elementString[i]) < 65:
                startIndex = i
                break
        
        li.text = elementString[startIndex:len(elementString)]
    elif boxType == "sublist":
        if newList:
            #move list text into <p> tags to be structure more clearly in line with
            #new sublist <ol> tags
            p = xml.SubElement(li, "p")
            p.text = li.text
            li.text = ""
            
            subOl = xml.SubElement(li, "ol")
            newList = False

        #find sublist identifier ( i), ii), etc. )
        for i in range(len(elementString)):
            if elementString[i] == ")":
                sublistNum = elementString[0:i]
                startIndex = i
                break

        subli = xml.SubElement(subOl, "li")
        subli.set("key", li.get("key")+"."+sublistNum)

        subli.text = elementString[startIndex:len(elementString)]


#Get XML data into elementTree and write to file
roughString = xml.tostring(root, 'utf-8')
reparsed = minidom.parseString(roughString)
prettyString = reparsed.toprettyxml(encoding='utf-8')

file = open("output.xml", "wb")
file.write(prettyString)
file.close()


#image is 4134x5847 scale down
#cv2.imshow('contourMassive', contour_disp)
for i in range(3):
    if (i == 2):
        cv2.imshow('groupedBig', grouped_disp)
        #cv2.imshow('contourBig', contour_disp)

    grouped_disp = cv2.pyrDown(grouped_disp)
    contour_disp = cv2.pyrDown(contour_disp)

#cv2.imshow('page', page_disp)   
#cv2.imshow('thresholded', thresh_disp)
#cv2.imshow('contoured', contour_disp)
cv2.imshow('grouped', grouped_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()



