#Grouping of contours for classification:
#to close opened windows, focus on any window and press ESC (or manually 'x' out of each one)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path

#CONVERTING PDF TO IMAGE
#pages = convert_from_path('NZBC-G4#3.4.pdf', 500)

#SAVING FIRST PAGE
#pages[6].save('page7.jpg', 'JPEG')

#threshold value (image binary thresholding)
threshold = 220

#distance threshold (contour association)
cDistance = 125

#read page, convert to grayscale and initial threshold
page = cv2.imread('page7.jpg')
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
cv2.drawContours(contour_disp, contours, -1, (0,0,255), 2)

xGroup = 0
yGroup = 0
endX = 0
endY = 0

xPrev = -1000
yPrev = -1000

listElement = 0

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)

    #ignore small contours (noise)
    if cv2.contourArea(contour) < 100:
        continue

    #determine if contour is within threshold, if not finalise previous
    #bounding box and draw
    if abs(x - xPrev) < cDistance or abs(y - yPrev) < cDistance:
        if x < xGroup:
            xGroup = x
        if y < yGroup:
            yGroup = y
        if x+w > endX:
            endX = x+w
        if y+h > endY:
            endY = y+h
    else:
        if xPrev != -1000 and yPrev != -1000:
            wGroup = endX - xGroup
            hGroup = endY - yGroup

            if wGroup*hGroup < contourOversize:
                cv2.rectangle(grouped_disp, (xGroup,yGroup),
                              (xGroup+wGroup,yGroup+hGroup),(0,0,255),2)

        xGroup = x
        yGroup = y
        endX = x+w
        endY = y+h
        
    xPrev = x
    yPrev = y


wGroup = endX - xGroup
hGroup = endY - yGroup
cv2.rectangle(grouped_disp, (xGroup,yGroup),(xGroup+wGroup,yGroup+hGroup),(0,0,255),2)
    
#image is 4134x5847 scale down
#for i in range(3):
#    page_disp = cv2.pyrDown(page_disp)
#    thresh_disp = cv2.pyrDown(thresh_disp)
#    contour_disp = cv2.pyrDown(contour_disp)

for i in range(2):
    contour_disp = cv2.pyrDown(contour_disp)
    grouped_disp = cv2.pyrDown(grouped_disp)

#cv2.imshow('page', page_disp)   
#cv2.imshow('thresholded', thresh_disp)
cv2.imshow('contoured', contour_disp)
cv2.imshow('grouped', grouped_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()



