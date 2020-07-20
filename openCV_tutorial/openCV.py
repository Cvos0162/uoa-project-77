#following tutorial at: https://www.youtube.com/watch?v=N81PCpADwKQ
#Warning: it's long ~= 9 & 1/2 hours
#Simplest stuff at the bottom, progressing upwards.

import cv2              #openCV
import datetime         #for datetime
import numpy as np
from matplotlib import pyplot as plt

#FOR CAM SHIFT which improves upon MEAN SHIFT method see last 6 minutes of
#tutorial video

#MEAN SHIFT OBJECT TRACKING
cap = cv2.VideoCapture('slow_traffic_small.mp4')

#take first frame of the video
ret, frame = cap.read()
#setup initial location of window (here it is the white car window)
x,y,width,height = 300, 200, 100, 50
track_window = (x,y,width,height)
#setup the ROI(region of interest) for tracking
roi = frame[y:y+height,x:x+width]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #convert to HSV

#produce histogram for back-projected image, histogram back-projection produces
#image of same size as original image where each pixel is a single channel
#containing the probability that the pixel belongs to our object
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180.,255.,255.)))

#only using hue channel hence [0] below
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#setup the termination criteria, either 10 iteration or move by at least 1 pt
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

#the disadvantage of this method as you have to know the initial
#poisition of the target object
while(1):
    ret,frame = cap.read()
    if ret == True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #get the back-projected image
        result = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        #apply meanshift to get the new location (of the object)
        ret, track_window = cv2.meanShift(result, track_window, term_criteria)

        #draw it on image
        x,y,w,h = track_window
        final_image = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 3)
        
        cv2.imshow('frame', frame)
        cv2.imshow('tracking', final_image)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


##BACKGROUND SUBTRACTION METHODS: producing mask containing binary image
##corresponding to moving elements of an image, i.e. a foreground mask to remove
##non-moving/static background
#cap = cv2.VideoCapture('vtest.avi')

##fgbg = foregroundBackground, first method
##fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
##Second method, by default detectShadows=True and are displayed asgrey color
##fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

##third method (results not as good as above two)
##fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
##kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

##fourth method (also does shadows with grey pixels detectShadows=True by
##default)
#fgbg = cv2.createBackgroundSubtractorKNN()

#while True:
#    ret, frame = cap.read()
#    if frame is None:
#        break

#    #applies the fgbg method to the frame to get the fgmask
#    fgmask = fgbg.apply(frame)

#    #required with GMG method to get results
#    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#    cv2.imshow('Frame', frame)
#    cv2.imshow('FG Masked', fgmask)

#    keyboard = cv2.waitKey(30)
#    if keyboard == 'q' or keyboard == 27:
#        break

#cap.release()
#cv2.destroyAllWindows()


##CORNER DETECTION using Shi Tomasi Corner Detector
##similar to Harris except R is calculated differently, results are improved
#img = cv2.imread('pic1.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##(image, maximum no. of corners to return (if more strongest 25 returned),
##quality level (minimal expected quality of corner), mimimum distance between
##returned corners)
#corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

##convert to integer (int0 is alias for int64)
#corners = np.int0(corners)

#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(img, (x,y), 3, 255, -1)

#cv2.imshow('dst', img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##CORNER DETECTION using Harris Corner Detector
##Harris Corner Detector:
##1. determine windows that produce very large variations in intensity
##2. compute an 'R' score for each window
##3. threshold these scores to find important corners
#img = cv2.imread('chessboard_img.png')

#cv2.imshow('img', img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gray = np.float32(gray)

##cornerHarris(image (in float32), blockSize (neighbourhood size), ksize
##(aperture paramter for sobel), k (harris detector free paramter))
#result = cv2.cornerHarris(gray, 2, 3, 0.04)

##dilate for better result
#result = cv2.dilate(result, None)

##marking corners with red color
#img[result > 0.01 * result.max()] = [0, 0, 255]

#cv2.imshow('result', img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##FACE DETECTION using Haar Cascade Classifiers (and EYE DETECTION)
##requires training, here we are using a pre-trained classifier from opencv
##github for faces and eyes
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

##can be done with an image
##img = cv2.imread('messi5.jpg')
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##or a video (frames)
#cap = cv2.VideoCapture('faceTest.mp4')

##remove while loop an cap.read if working with single image.
#while cap.isOpened():
#    _,img = cap.read()

#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    #detectMultiScale(image, scaleFactor, minNeighbours)
#    #scaleFactor = how much the image size is reduced at each image scale
#    #minNeighbours = how many neighbours each candidate rectangle should have
#    #to retain it
#    #produces an array of rectangles where faces are found in the image
#    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#    for (x,y,w,h) in faces:
#        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

#        #eyes are present on faces so we apply the eye classifer to this region
#        #roi = region of interest
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = img[y:y+h, x:x+w]
#        eyes = eye_cascade.detectMultiScale(roi_gray)
#        for (ex, ey, ew, eh) in eyes:
#            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 5)

#    cv2.imshow('img', img)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
##cv2.waitKey(0) #add back if working with single image
#cv2.destroyAllWindows()


##CIRCLE DETECTION using hough circle transform
##img = cv2.imread('smarties.png')
##output = img.copy()

##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##gray = cv2.medianBlur(gray, 5)

##HoughCircles(source image, method, ratio of output resolution to input
##resolution, min distance between detected circles, first method-specific
##parameter, second method-specific parameter, min radius, max radius)
##NOTE: only implemented method is HOUGH_GRADIENT
##param1 in this method is higher threshold of the two passed to canny edge
##detector
##param2 in this method is the threshold value for circle centers for detection
##maxRadius >= 0 maximum image dimension is used, if < 0 returns centers without
##radius
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50,
#                           param2=30, minRadius=0, maxRadius=0)
#detected_circles = np.uint16(np.around(circles))
#for (x,y,r) in detected_circles[0, :]:
#    cv2.circle(output, (x,y), r, (0,255,0), 3) #draw circle
#    cv2.circle(output, (x,y), 2, (0,255,255), 3) #drawing a dot at center

#cv2.imshow('output', output)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##VIDEO ROAD LANE DETECTION
#def region_of_interest(img, vertices):
#    mask = np.zeros_like(img)
#    match_mask_color = 255
#    cv2.fillPoly(mask, vertices, match_mask_color)
#    masked_image = cv2.bitwise_and(img, mask)
#    return masked_image

#def drawLines(img, lines):
#    img = np.copy(img)
#    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

#    for line in lines:
#        for x1, y1, x2, y2 in line:
#            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), thickness=10)

#    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
#    return img

#def process(image):
#    height = image.shape[0]
#    width = image.shape[1]
#    region_of_interest_vertices = [
#        (0, height),
#        (width/2, height/2),
#        (width, height)
#    ]

#    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#    canny_image = cv2.Canny(gray_image, 100, 120)
#    cropped_image = region_of_interest(canny_image,
#                                       np.array([region_of_interest_vertices],
#                                                np.int32),)
#    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50,
#                            lines=np.array([]),
#                            minLineLength=40,
#                            maxLineGap=100)
#    image_with_lines = drawLines(image, lines)
#    return image_with_lines

#cap = cv2.VideoCapture('test.mp4')

#while cap.isOpened():
#    ret,frame = cap.read()
#    frame = process(frame)
#    cv2.imshow('video', frame)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()


##IMAGE ROAD LANE DETECTION using hough transform etc
#image = cv2.imread('road.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#print(image.shape)
#height = image.shape[0]
#width = image.shape[1]

##region of interest ignores parts of the image that are not road
#region_of_interest_vertices = [
#    (0, height),
#    (width/2, height/2),
#    (width, height)
#]

##function to mask area outside region of interest
#def region_of_interest(img, vertices):
#    mask = np.zeros_like(img)
#    match_mask_color = 255
#    cv2.fillPoly(mask, vertices, match_mask_color)
#    masked_image = cv2.bitwise_and(img, mask)
#    return masked_image

#def drawLines(img, lines):
#    img = np.copy(img)
#    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

#    for line in lines:
#        for x1, y1, x2, y2 in line:
#            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), thickness=3)

#    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
#    return img

##want to perform edge detection(canny) before applying the mask otherwise
##the edge of the mask is detected as lines on the image which we don't want
#gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#canny_image = cv2.Canny(gray_image, 100, 200)
#cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices],
#                                                   np.int32),)

#lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/60, threshold=160,
#                        lines=np.array([]), minLineLength=40, maxLineGap=25)
#image_with_lines = drawLines(image, lines)

#plt.imshow(image_with_lines)
#plt.show()


##PROBABILISTIC HOUGH TRANSFORM to detect lines
#img = cv2.imread('sudoku.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)

##HoughLinesP(edge detected image, pixel resolution, angle resolution, threshold,
##minimum length of a line (shorter lines rejected), maximum gap between lines
##(line segments with gaps smaller than this are treated as a single line)
##HoughLinesP returns nice direct points for the line vs polar coords from
##HoughLines
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

#for line in lines:
#    x1,y1,x2,y2 = line[0]
#    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

#cv2.imshow('edges', edges)
#cv2.imshow('image', img)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()


##THE HOUGH TRANSFORM: a popular technique to detect any shape that can
##be represented in a mathematical form even if the shape is broken or distorted

##Hough Transform basics:
##- a line in an image can be expressed with two variables, y = mx + c or polar
##  xcos() + ysin() = r
##- in xy space we have y = mx + c where m = slope, c = intercept
##  in 'hough space' or 'mc space' m is like x (horizontal axis and c is
##  like y (vertical axis)
##  the opposite of this is also true, a point (x,y) in xy space can be
##  represented as a line in the mc space c = -xm + y
##- Therefore hough transform is about converting an xy line to hough space
##  point or xy point to mc line
##- y = mx + c is not able to represent vertical lines so we use the polar
##  representation in the algorithm and applied methods, for more info on
##  polar see documentation/tutorials online

##The Hough transformation Algorithm
##1. Edge detection, e.g. using the canny edge detector
##2. Mapping of edge points to the Hough space and storage in an accumulator
##3. Interpretation of the accumulator to yield lines of infinite length; done by
##   thresholding/other constraints
##4. Conversion of infinite lines to finite lines
##Implemented in two methods in openCV:
##Standard: HoughLines method
##Probabilisitc: HoughLinesP method

##Using standard hough Transform to detect lines
##the problem with this is the lines are infinite and will run from the edge
##of the image to the other; fixed with probabilisitc method above this
#img = cv2.imread('sudoku.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray, 50, 150, apertureSize=3)

##HoughLines(source image, rho; distance resolution in pixels, theta: angle
##resolution in radians, threshold for lines)
##NOTE: each line is in polar coords
#lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

#for line in lines:
#    rho,theta = line[0]
#    a = np.cos(theta)
#    b = np.sin(theta)

#    #x0,y0 is the top left of the image
#    x0 = a*rho
#    y0 = b*rho

#    #x1 stores the rounded off value of (r*cos(theta)-1000*sin(theta))
#    x1 = int(x0 + 1000 * (-b))
#    #y1 stores the rounded off value of (r*sin(theta)+1000*cos(theta))
#    y1 = int(y0 + 1000 * (a))
#    #x2 stores the rounded off value of (r*cos(theta)+1000*sin(theta))
#    x2 = int(x0 - 1000 * (-b))
#    #y2 stores the rounded off value of (r*sin(theta)-1000*cos(theta))
#    y2 = int(y0 - 1000 * (a))

#    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

#cv2.imshow('canny', edges)
#cv2.imshow('image', img)
#k = cv2.waitKey(0)
#cv2.destroyAllWindows()


##TEMPLATE MATCHING: finding a template image inside a larger image
#img = cv2.imread('messi5.jpg')
#grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#template = cv2.imread('messi_face.jpg', 0)
#w,h = template.shape[::-1] #grab values in reverse order to get w,h

##in result will be stored an array with values corresponding to the match
##between the source image and the template, the higher the number the closer
##the match
##matchTemplate(source, template, method)
#result = cv2.matchTemplate(grey_img, template, cv2.TM_CCOEFF_NORMED)
#threshold = 0.99
#loc = np.where(result >= threshold) #gives values from matrix meeting condition
#for pt in zip(*loc[::-1]): #zip as each loc element is x,y
#    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,0,255), 2)

#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##IMAGE HISTOGRAMS: histogram graphs the intensity distribution of an image
##tells you about lighting conditions etc.
#img = cv2.imread('lena.jpg', 0)

##for a simple shape image
##img = np.zeros((200,200), np.uint8)
##cv2.rectangle(img, (0,100), (200,200), (255), -1)
##cv2.rectangle(img, (0,50), (100,100), (127), -1)

##histogram using matplotlib, extra stuff here for blue/green/red plots
##b,g,r = cv2.split(img)
##cv2.imshow('img', img)
##cv2.imshow('b', b)
##cv2.imshow('g', g)
##cv2.imshow('r', r)
##hist(image, max number of pixel values, range)
##plt.hist(img.ravel(), 256, [0, 256])
##plt.hist(b.ravel(), 256, [0, 256])
##plt.hist(g.ravel(), 256, [0, 256])
##plt.hist(r.ravel(), 256, [0, 256])
##plt.show()

##histogram using cv2
##calcHist(list of images, channels (here we are just in greyscale), mask, ...
##... hist size, range)
#hist = cv2.calcHist([img], [0], None, [256], [0, 256])
#plt.plot(hist)
#plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##SIMPLE SHAPE DETECTION
#img = cv2.imread('shapes.jpg')
#imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#_,thresh = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
#contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#for contour in contours:
#    #method approximates polygon curve
#    #approxPolyDP(contour, epsilon (accuracy), closed or open shape)
#    #arcLength(contour, open or closed shape)
#    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
#    cv2.drawContours(img, [approx], 0, (0,0,0), 5)
#    x = approx.ravel()[0]
#    y = approx.ravel()[1]

#    #three curves is likely a triangle, similar logic for rest
#    if len(approx) == 3:
#        cv2.putText(img, 'Triangle', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                    (0,0,0))
#    elif len(approx) == 4:
#        x,y,w,h = cv2.boundingRect(approx)
#        aspectRatio = float(w)/h

#        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
#            cv2.putText(img, 'Square', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                    (0,0,0))
#        else:
#            cv2.putText(img, 'Rectangle', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                    (0,0,0))
#    elif len(approx) == 5:
#        cv2.putText(img, 'Pentagon', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                    (0,0,0))
#    elif len(approx) == 10:
#        cv2.putText(img, 'Star', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
#    else:
#        cv2.putText(img, 'Circle', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,0))

#cv2.imshow('shapes', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##SIMPLE MOTION DETECTION/TRACKING
#cap = cv2.VideoCapture('vtest.avi')

#ret, frame1 = cap.read()
#ret, frame2 = cap.read()

#while cap.isOpened():
#    diff = cv2.absdiff(frame1, frame2)
    
#    #finding contours in grayscale is easier
#    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#    #GaussianBlur(source, ksize, sigmaX)
#    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
#    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#    dilated = cv2.dilate(thresh, None, iterations=3) #providing no/'None' kernel
#    contours,_ = cv2.findContours(dilated, cv2.RETR_TREE,
#                                  cv2.CHAIN_APPROX_SIMPLE)
    
#    #cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
#    for contour in contours:
#        (x, y, w, h) = cv2.boundingRect(contour)

#        #adjusting this value to ignore noise
#        if cv2.contourArea(contour) < 900:
#            continue

#        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)

#        #writing text if any movement detected
#        cv2.putText(frame1, 'Status: {}'.format('Movement'), (10, 20),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        
#    cv2.imshow('feed', frame1)

#    #getting the next two frames for comparison
#    frame1 = frame2
#    ret, frame2 = cap.read()
#
#    if cv2.waitKey(40) == 27:
#        break

#cv2.destroyAllWindows()
#cap.release()


##FINDING AND DRAWING CONTOURS: a contour is a curve of pixels of the same
##color or intensity along a boundary
#img = cv2.imread('opencv-logo.png')
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##threshold(source, threshold, maximum, type)
#ret, thresh = cv2.threshold(imgray, 127, 255, 0)

##findContours(thresholded image, contour retrieval mode, contour approx method)
##contours is a python list of all contours in the image, each
##contour is a numpy array of (x,y) coordinates of boundary points
#contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,
#                                      cv2.CHAIN_APPROX_NONE)
#print('Number of contours = '+str(len(contours)))

##drawContours(original image, contours, contour indexes(-1 draws all), color, thickness)
#cv2.drawContours(img, contours, -1, (0,255,0), 3)

##could instead only draw one contour if desired (number must be inside list of
##contours i.e. list of 9 max index num is 8)
##cv2.drawContours(img, contours, 4, (0,255,0), 3)

#cv2.imshow('image', img)
#cv2.imshow('image GRAY', imgray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##IMAGE BLENDING using image pyramids
#apple = cv2.imread('apple.jpg')
#orange = cv2.imread('orange.jpg')
#print(apple.shape) #shape = size (width, height, depth)
#print(orange.shape)

##straight combining left half apple with right half orange
#apple_orange = np.hstack((apple[:, :256], orange[:, 256:]))

##now image blending can be done in 5 steps:
##1. load two images
##2. find gaussian pyramids for both (with a certain number of pyramid levels)
##3. find laplacian pyramids from gaussian pyramid
##4. for this example we are joining left half of apple with right half of
##   orange in each level of the laplacian pyramid
##5. from these joint image pyramid we can reconstruct the blended image

##generate gaussian pyramid for apple
#apple_copy = apple.copy()
#gp_apple = [apple_copy]
#for i in range(6):
#    apple_copy = cv2.pyrDown(apple_copy)
#    gp_apple.append(apple_copy)

##generate gaussian pyramid for orange
#orange_copy = orange.copy()
#gp_orange = [orange_copy]
#for i in range(6):
#    orange_copy = cv2.pyrDown(orange_copy)
#    gp_orange.append(orange_copy)

##generate laplacian pyramid for apple
#apple_copy = gp_apple[5]
#lp_apple = [apple_copy]
#for i in range(5, 0, -1):
#    gaussian_extended = cv2.pyrUp(gp_apple[i])
#    laplacian = cv2.subtract(gp_apple[i-1], gaussian_extended)
#    lp_apple.append(laplacian)

##generate laplacian pyramid for orange
#orange_copy = gp_orange[5]
#lp_orange = [orange_copy]
#for i in range(5, 0, -1):
#    gaussian_extended = cv2.pyrUp(gp_orange[i])
#    laplacian = cv2.subtract(gp_orange[i-1], gaussian_extended)
#    lp_orange.append(laplacian)

##now add left and right halves of images in each level
#apple_orange_pyramid = []
#n = 0
#for apple_lap, orange_lap in zip(lp_apple, lp_orange):
#    n += 1
#    cols, rows, channels = apple_lap.shape
#    laplacian = np.hstack((apple_lap[:, 0:int(cols/2)],
#                           orange_lap[:, int(cols/2):]))
#    apple_orange_pyramid.append(laplacian)

##now reconstruct
#apple_orange_reconstruct = apple_orange_pyramid[0]
#for i in range(1, 6): #default step-size is 1
#    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
#    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i],
#                                       apple_orange_reconstruct)
    
    
#cv2.imshow('apple', apple)
#cv2.imshow('orange', orange)
#cv2.imshow('unblended apple-orange', apple_orange)
#cv2.imshow('blended apple_orange', apple_orange_reconstruct)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##IMAGE PYRAMIDS: images of different resolutions, i.e. the first level is
##a search of the original image, the second level is a search of the image
##at 1/2 resolution (after blurring, smoothing, etc..) and so on (1/4, 1/8 ...)
##Two types: Gaussian pyramid, Laplacian pyramid
#img = cv2.imread('lena.jpg')

##lr = lower resolution, pyrDown is the first method of Gaussian pyramid
##hr = higher resolution, pyrUp second method of Gaussian pyramid
##lr1 = cv2.pyrDown(img)
##lr2 = cv2.pyrDown(lr1)
##hr2 = cv2.pyrUp(lr2)

##alternative to above repeated pyrDown/Up for gaussian pyramid
#layer = img.copy()
#gp = [layer] #gp = gaussian pyramid
#for i in range(6):
#    layer = cv2.pyrDown(layer)
#    gp.append(layer)
#    #cv2.imshow(str(i), layer)

##there is no direct function/method for a laplacian pyramid
##instead a level in the laplacian pyramid is formed by the
##difference between that level in the gaussian pyramid and
##the expanded version of its upper level in gaussian pyramid
#layer = gp[5] #last level of gaussian
#cv2.imshow('upper level Gaussian Pyramid', layer)
#lp = [layer] #lp = laplacian pyramid

##for loop 5 -> 1 (doesn't include 0)
#for i in range(5, 0, -1):
#    gaussian_extended = cv2.pyrUp(gp[i])
#    laplacian = cv2.subtract(gp[i-1], gaussian_extended)
#    cv2.imshow(str(i), laplacian)

#cv2.imshow('Original image', img)
##cv2.imshow('pyrDown 1', lr1)
##cv2.imshow('pyrDown 2', lr2)
##cv2.imshow('pyrUp 1', hr2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##CANNY EDGE DETECTOR: edge detection operator that uses multi-stage algorithm to
##detect a 'wide range of edges in images'
##Steps:
##1. Noise reduction
##2. Gradient calculation
##3. Non-maximum suppression - get rid of 'spurious' response
##4. Double threshold
##5. Edge tracking by Hysteresis
#img = cv2.imread('messi5.jpg', 0)

##Canny(img source, 1st threshold, 2nd threshold) 
##threshold values are for the hysteresis step
#canny = cv2.Canny(img, 100, 200)

#titles = ['image', 'Canny']
#images = [img, canny]

#for i in range(2):
#    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])

#plt.show()


##IMAGE GRAIDENTS: a directional change in the intensity or color in an image
##Used for edge detection
#img = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)

##Laplacian(source image, datatype, kernal size)
##64F = 64-bit float (supports negative numbers used in laplacian method)
#lap = cv2.Laplacian(img, cv2.CV_64F, ksize=1)

##convert values back to uint8 for RGB img
##NOTE: negative numbers require absolute for uint8
#lap = np.uint8(np.absolute(lap))

##Sobel(source image, datatype, dx, dy)
##where dx = order of derivative x, dy = order of derivative y
##5th argument (ksize=x) is possible similar to laplacian
#sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
#sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

#sobelX = np.uint8(np.absolute(sobelX))
#sobelY = np.uint8(np.absolute(sobelY))

#sobelCombined = cv2.bitwise_or(sobelX, sobelY)

#titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined']
#images = [img, lap, sobelX, sobelY, sobelCombined]

#for i in range(5):
#    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])

#plt.show()


##SMOOTHING IMAGES and BLURRING IMAGES: used to remove noise etc#####
#img = cv2.imread('lena.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#kernel = np.ones((5, 5), np.float32)/25 #kernel = 1/Kwidth*Kheight * K

##homogenous filter: each pixel is the mean of kernel neighbour
##filter2D(source image, desired depth of dest, kernel)
#destination = cv2.filter2D(img, -1, kernel)
#blur = cv2.blur(img, (5,5))

##gaussian filter is similar to homogenous but rather than a e.g. 5x5
##matrix of equally weighted values i.e. all 1's you have different values
##that is weight decreases across the kernal as you get further from the
##target pixel (middle value)
#gblur = cv2.GaussianBlur(img, (5,5), 0) #second argument is the kernel

##median filter replaces each pixel with the median of its neighbouring pixels
##good for 'salt and pepper' noise
#median = cv2.medianBlur(img, 5) #kernel size must be odd (but not 1x1)

##above methods blur the edges but sometimes we want to retain the sharp edges
##(img, neighbour/kernel size, sigma space
#bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

#titles = ['image', '2D Convolution', 'blur', 'GaussianBlur', 'median',
#          'bilateralFilter']
#images = [img, destination, blur, gblur, median, bilateralFilter]

#for i in range(6):
#    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])

#plt.show()


##MORPHOLOGICAL TRANSFORMATIONS: transformations based on shape, normally
##performed on binary images

##loading a normal image (if you load a binary image no need to mask it)
#img = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
#_,mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

##kernel: a square or some shape we 'apply' on the image
##here we are attempting to remove the spots present in the masked smarties
#kernal = np.ones((2,2), np.uint8) #a 2x2 square shape kernel

##effectively, dilation sets pixels adjacent to a '1' to '1' hence 'dilating'
##the area
#dilation = cv2.dilate(mask, kernal, iterations=2)

##erosion is the opposite of dilation, where pixels adjacent to '0' are set to
##'0' hence eroding at the edges
#erosion = cv2.erode(mask, kernal, iterations=1)

##opening is erosion followed by dilation
#opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

##closing is dilation followed by erosion
#closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

##these methods and various others see documentation for more info
#mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)
#th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernal)

#titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg',
#          'th']
#images = [img, mask, dilation, erosion, opening, closing, mg, th]

#for i in range(8):
#    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])

#plt.show()


##MULTIPLE IMAGES IN SINGLE WINDOW on a single windows (subplots in matplotlib)
#img = cv2.imread('gradient.png', 0)
#_,th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
#_,th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
#_,th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
#_,th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
#_,th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

#titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO',
#          'TOZERO_INV']
#images = [img, th1, th2, th3, th4, th5]

#for i in range(6):
#    #subplot(rows, columns, index of image)
#    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]), plt.yticks([])

#plt.show()


##MATPLOTLIB with OPENCV: 2D plotting library for python#####
##advantage of MATPLOTLIB: multiple GUI options for interacting with plot
#img = cv2.imread('lena.jpg', -1)
#cv2.imshow('image', img)

##opening image using matplotlib window
##NOTE: matplotlib uses RGB where opencv uses BGR
##convert BGR to RGB
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.xticks([]), plt.yticks([]) #hides axis 'ticks' in plot window
#plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##ADAPTIVE THRESHOLDING#####
##as opposed to simple thresholding this is adaptive thresholding where
##the thresholding is calculated for smaller regions of the same image
##used to account for lighting etc (images with varying illumination)
#img = cv2.imread('sudoku.png', 0)
#_,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) #simple thresholding

##(source image, maximum value for a pixel (255 for us), adaptive method, ...
##... threshold type, block size (i.e. 11x11), value of c (constant that is
##subtracted from the mean)
#th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                               cv2.THRESH_BINARY, 11, 2)
#th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv2.THRESH_BINARY, 11, 2)

#cv2.imshow('image', img)
#cv2.imshow('th1', th1)
#cv2.imshow('th2', th2)
#cv2.imshow('th3', th3)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##SIMPLE THRESHOLDING: seperating an object from its background#####
#img = cv2.imread('gradient.png',0)

#below is simple thresholding where a global threshold value is used for
#every pixel (the second value in the function)
##(source image, threshold value, maximum threshold value, threshold type)
##Binary thresholding: 1 if above 0 if below
#_,th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
#_,th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV) #inverse binary

##upto threshold value is unchanged, beyond threshold pixel value is set to the
##threshold.
#_,th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

##value less than threshold is assigned to 0
#_,th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
#_,th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV) #inverse tozero

#cv2.imshow('Image', img)
#cv2.imshow('th1', th1)
#cv2.imshow('th2', th2)
#cv2.imshow('th3', th3)
#cv2.imshow('th4', th4)
#cv2.imshow('th5', th5)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##HUE, SATURATION and VALUE for image tracking (detecting specific color(s)
##within an image#####
#def nothing(x):
#    pass

#cv2.namedWindow('Tracking')
#cv2.createTrackbar('LH', 'Tracking', 0, 255, nothing) #lower hue value
#cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing) #lower sat value
#cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing) #lower val value
#cv2.createTrackbar('UH', 'Tracking', 255, 255, nothing) #upper hue value
#cv2.createTrackbar('US', 'Tracking', 255, 255, nothing) #upper sat value
#cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing) #upper val value

#while True:
#    #NOTE: can do same with a camera using videoCapture
#    #frame = cap.read()
#    frame = cv2.imread('smarties.png')

#    #convert BGR to HSV
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#    #upper and lower range for hsv
#    l_h = cv2.getTrackbarPos('LH', 'Tracking')
#    l_s = cv2.getTrackbarPos('LS', 'Tracking')
#    l_v = cv2.getTrackbarPos('LV', 'Tracking')
#    u_h = cv2.getTrackbarPos('UH', 'Tracking')
#    u_s = cv2.getTrackbarPos('US', 'Tracking')
#    u_v = cv2.getTrackbarPos('UV', 'Tracking')
    
#    #lower and upper range for color
#    l_b = np.array([l_h,l_s,l_v])
#    u_b = np.array([u_h,u_s,u_v])

#    #mask effectively removes all colors outside specified range
#    mask = cv2.inRange(hsv, l_b, u_b)
#    res = cv2.bitwise_and(frame,frame,mask=mask)

#    cv2.imshow('frame', frame)
#    cv2.imshow('mask', mask)
#    cv2.imshow('result', res)

#    key = cv2.waitKey(1)
#    if key == 27:
#        break

#cv2.destroyAllWindows()


##TRACKBARS and SWITCHES(type of trackbar)#####
#def nothing(x):
#        pass

#img = np.zeros((300,512,3), np.uint8)
#cv2.namedWindow('image')

##(name of trackbar, name of window, inital value, final value, callback function)
##NOTE: callback function called on trackbar value changing.
#cv2.createTrackbar('B', 'image', 0, 255, nothing)
#cv2.createTrackbar('G', 'image', 0, 255, nothing)
#cv2.createTrackbar('R', 'image', 0, 255, nothing)

#switch = '0 : OFF\n 1 : ON'
#cv2.createTrackbar(switch, 'image', 0, 1, nothing)

#while(1):
#        cv2.imshow('image',img)

#        #same as previous code break if esc pushed
#        k = cv2.waitKey(1) & 0xFF
#        if k == 27:
#                break

#        #use trackbar to set the image color
#        b = cv2.getTrackbarPos('B', 'image')
#        g = cv2.getTrackbarPos('G', 'image')
#        r = cv2.getTrackbarPos('R', 'image')
#        s = cv2.getTrackbarPos(switch, 'image')

#        if s == 0:
#                img[:] = 0
#        else:
#                img[:] = [b,g,r]
        
#cv2.destroyAllWindows()


##BITWISE operations on images#####
#img1 = np.zeros((250, 500, 3), np.uint8)
#img1 = cv2.rectangle(img1,(200,0), (300,100), (255,255,255), -1) #white rectangle on black image above
#img2 = cv2.imread('image_1.png')

#bitAnd = cv2.bitwise_and(img2, img1)
#bitOr = cv2.bitwise_or(img2, img1)
#bitXor = cv2.bitwise_xor(img2, img1)
#bitNot1 = cv2.bitwise_not(img1)
#bitNot2 = cv2.bitwise_not(img2)

#cv2.imshow('img1', img1)
#cv2.imshow('img2', img2)
#cv2.imshow('not1', bitNot1)
#cv2.imshow('not2', bitNot2)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##IMAGE PROPERTIES; shape, size channels + copying portions of an image, adding images etc#####
#img = cv2.imread('messi5.jpg')
#img2 = cv2.imread('opencv-logo.png')

#print(img.shape)        #returns a tuple of number or rows, columns and channels
#print(img.size)         #returns total number of pixels accessed
#print(img.dtype)        #returns image datatype
#b,g,r = cv2.split(img)  #split an image into its individual channels
#img = cv2.merge((b,g,r))#merge bgr channels into an image

##so taking ball coords and setting image pixels at another location to the ball pixels
#ball = img[280:340, 330:390]
#img[273:333, 100:160] = ball

##adding pixel values of the two images for an interesting result
##NOTE: images(arrays) must be the same size so we have to resize them first
#img = cv2.resize(img, (512,512))
#img2 = cv2.resize(img2, (512,512))

##straight just add the two images
##result = cv2.add(img, img2)

##instead add weighted where one image can be weighted above the other
##(firstImage, weight1, secondImage, weight2, scalar to add to pixels)
#result = cv2.addWeighted(img, 0.9, img2, 0.1, 0)

#cv2.imshow('image', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##MOUSE events to draw a line#####
#def click_event(event, x, y, flags, param):
#        if event == cv2.EVENT_LBUTTONDOWN:
#                ##draw a small point(circle) at the clicked location
#                #cv2.circle(img, (x,y), 3, (0,0,255),-1)

#                ##when we click we save the points, using the last two clicked points we draw
#                ##a line between the two
#                #points.append((x,y))
#                #if len(points) >= 2:
#                #        cv2.line(img, points[-1], points[-2], (255,0,0), 5)

#                #taking color clicked on one image and putting onto another image
#                blue = img[x, y, 0]
#                green = img[x, y, 1]
#                red = img[x, y, 2]
#                cv2.circle(img, (x,y), 3, (0,0,255), -1)
#                myColorImage = np.zeros((512,512,3), np.uint8)

#                myColorImage[:] = [blue, green, red]
                
#                cv2.imshow('color', myColorImage)

##img = np.zeros((512,512,3), np.uint8)
#img = cv2.imread('lena.jpg')
#cv2.imshow('image', img)

#points = []

#cv2.setMouseCallback('image', click_event)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


##MOUSE events#####
##code below to show all the available events
##dir shows all function names/events available in the 'cv2' package
##events = [i for i in dir(cv2) if 'EVENT' in i]
##print(events)

##callback function for mouse pressing
#def click_event(event, x, y, flags, param):
#        #for left clicking we are putting the clicked x,y text at the x,y on the image
#        if event == cv2.EVENT_LBUTTONDOWN:
#                strXY = str(x) + ', '+ str(y)
                
#                font = cv2.FONT_HERSHEY_SIMPLEX
#                cv2.putText(img, strXY, (x,y), font, 1, (0,255,255), 2)
#                cv2.imshow('image', img)

#        #for right clicking we are pulling the bgr of the position clicked
#        if event == cv2.EVENT_RBUTTONDOWN:
#                blue = img[y, x, 0]
#                green = img[y, x, 1]
#                red = img[y, x, 2]

#                font = cv2.FONT_HERSHEY_SIMPLEX
#                colors = str(blue)+', '+str(green)+', '+str(red)
#                cv2.putText(img, colors, (x,y), font, 1, (255,255,0), 2)
#                cv2.imshow('image', img)

##black image
##img = np.zeros((512,512,3), np.uint8)

##load image
#img = cv2.imread('lena.jpg')
#cv2.imshow('image', img)

#cv2.setMouseCallback('image', click_event)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##TIME/RUNNING/PRINTING onto video#####
#cap = cv2.VideoCapture('vtest.avi')

#while(cap.isOpened()):
#        ret, frame = cap.read()
	
#        if ret:
#                font = cv2.FONT_HERSHEY_SIMPLEX

#                #generic string for putting
#                text = 'Width: '+str(cap.get(3)) + ' Height: ' + str(cap.get(4))

#                #getting datetime (and convert to string)
#                datet = str(datetime.datetime.now())

#                #NOTE: you can also put any shapes you wish using code from DRAWING below.
#                #(frame, text, position, font, fontscale, color, thickness, linetype)
#                frame = cv2.putText(frame, datet, (10,50), font, 1, (255,0,0), 2, cv2.LINE_AA)
#                cv2.imshow('frame', frame)
		
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
		
#cap.release()
#cv2.destroyAllWindows()


##DRAWING onto image#####
##reading as before
##img = cv2.imread('lena.jpg', 1)

##creating an image instead using numpy (a black image)
##array is [height, width, depth?]
#img = np.zeros([512, 512, 3], np.uint8)

##line parameters: (file, start, end, BGR color, thickness)
#img = cv2.line(img,(0,0),(255,255),(0,255,0),5)
#img = cv2.arrowedLine(img,(0,300),(300,300),(0,255,0),5)

##rectangle parameters: (file, top left, bot right, BGR, thickness)
##using thickness -1, fills the rectangle with color.
#img = cv2.rectangle(img,(384,0),(510,128),(0,0,255),5)

##circle parameters: (file, center, radius, BGR, thickness)
#img = cv2.circle(img,(447,63),63,(255,0,0),-1)

##text parameters: (file, text, start, fontFace, fontSize, BGR, thickness, ...
##... lineType)
#font = cv2.FONT_HERSHEY_SIMPLEX
#img = cv2.putText(img,'OpenCV',(10,500),font,4,(255,255,255),10,cv2.LINE_AA)

#cv2.imshow('image', img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


##CAPTURING/WRITING video#####
#cap = cv2.VideoCapture('vtest.avi')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

##VideoWriter parameters: (file, fourcc code, frame-rate, frame size)
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (768,576))

##size gotten from:
#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#while(cap.isOpened()):
#    ret, frame = cap.read()

#    #True if frame captured
#    if ret:
#        cv2.imshow('video', frame)

#        out.write(frame)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#out.release()
#cv2.destroyAllWindows()


##READING/WRITING to image file#####
#img = cv2.imread('lena.jpg', -1) #0=grayscale, 1=color, -1=auto
#cv2.imshow('image', img)

##Hold image until closed by user
#key = cv2.waitKey(0)

##escape key press = 27
#if key == 27:
#    cv2.destroyAllWindows()
#elif key == ord('s'):
#    cv2.imwrite('lena_saveCopy.png', img) #WRITE IMAGE
#    cv2.destroyAllWindows()
