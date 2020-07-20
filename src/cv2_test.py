#Thresholding and contouring:
#ensure the page1.jpg file is in the same folder as this script
#to close opened windows, focus on any window and press ESC (or manually 'x' out of each one)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path

#CONVERTING PDF TO IMAGE
#pages = convert_from_path('../sample/NZBC-E2#3.8.pdf', 500)

#SAVING FIRST PAGE
#pages[0].save('../output/page1.jpg', 'JPEG')

#global threshold value
threshold = 200
prevThresh = 200

#null callback function
def nothing(x):
    pass

cv2.namedWindow('thresholded')
cv2.createTrackbar('Threshold', 'thresholded', 0, 255, nothing)

#read page, convert to grayscale and initial threshold
page1 = cv2.imread('../output/page1.jpg')
page1_gray = cv2.cvtColor(page1, cv2.COLOR_BGR2GRAY)
ret, page1_thresh = cv2.threshold(page1_gray, 220, 255, cv2.THRESH_BINARY)

#image is 4134x5847 scale down
page1_disp = page1
gray_disp = page1_gray
for i in range(3):
    page1_disp = cv2.pyrDown(page1_disp)
    gray_disp = cv2.pyrDown(gray_disp)

cv2.imshow('page', page1_disp)
cv2.imshow('gray', gray_disp)

#loop allows for trackbar adjustment of threshold value
while True:
    threshold = cv2.getTrackbarPos('Threshold', 'thresholded')
    ret, page1_thresh = cv2.threshold(page1_gray, threshold, 255,
                                      cv2.THRESH_BINARY)

    contour_disp = page1
    if prevThresh != threshold:
        #find contours
        contours,_ = cv2.findContours(page1_thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contour_disp, contours, -1, (0,0,255), 3)
    
    #scaling down for threshold & contour images
    thresh_disp = page1_thresh
    for i in range(3):
        thresh_disp = cv2.pyrDown(thresh_disp)
        contour_disp = cv2.pyrDown(contour_disp)
        
    cv2.imshow('thresholded', thresh_disp)
    cv2.imshow('contoured', contour_disp)

    prevThresh = threshold

    #exit on ESC
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()



