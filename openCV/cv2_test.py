#Grouping of contours for classification:
#to close opened windows, focus on any window and press ESC (or manually 'x' out of each one)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path

#CONVERTING PDF TO IMAGE
#pages = convert_from_path('NZBC-G4#3.4.pdf', 500)

#SAVING FIRST PAGE
#pages[20].save('page20.jpg', 'JPEG')

#threshold value (image binary thresholding)
threshold = 200

#distance thresholds (contour association)
cDistance = 150

#read page, convert to grayscale and initial threshold
page = cv2.imread('page20.jpg')
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

elementCount = 0

while len(contours) > 1:
    xStart = pageWidth-1
    xEnd = 0
    yStart = pageHeight-1
    yEnd = 0

    for j in range(3):
        for i in range(len(contours)):
            contour = contours[i]
            (x,y,w,h) = cv2.boundingRect(contour)

            #ignore large contours (page border)
            if cv2.contourArea(contour) > contourOversize:
                continue

            cv2.rectangle(grouped_disp, (x,y),(x+w,y+h),(255,0,0),1)

            if xEnd == 0 or yEnd == 0:
                xStart = x
                yStart = y
                xEnd = x+w
                yEnd = y+h
                
            if x < xEnd+cDistance and x > xStart-cDistance and y < yEnd+cDistance and y > yStart-cDistance:
                cv2.rectangle(grouped_disp, (x,y),(x+w,y+h),(0,255,0),1)

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

    elementCount += 1
    cv2.rectangle(grouped_disp, (xStart,yStart),(xEnd,yEnd),(0,0,255),2)
    #remove contours inside element box from list
    for i in range(len(contours)-1,-1,-1):
        contour = contours[i]
        (x,y,w,h) = cv2.boundingRect(contour)

        if (x < xEnd and x >= xStart) and (y < yEnd and y >= yStart):
            del contours[i]
            #cv2.rectangle(grouped_disp, (x,y),(x+w,y+h),(255,0,0),1)

print(elementCount)
#image is 4134x5847 scale down
for i in range(3):
    #thresh_disp = cv2.pyrDown(thresh_disp)
    contour_disp = cv2.pyrDown(contour_disp)
    grouped_disp = cv2.pyrDown(grouped_disp)

#cv2.imshow('page', page_disp)   
#cv2.imshow('thresholded', thresh_disp)
cv2.imshow('contoured', contour_disp)
cv2.imshow('grouped', grouped_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()



