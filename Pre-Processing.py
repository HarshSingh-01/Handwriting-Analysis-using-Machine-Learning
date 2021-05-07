# This script fot testing the inidividual images.
# Real code will for preprocessing and feature etraction will be in 

import cv2
import numpy as np
import math

# Some temp variables that are not needed
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000

# Features are defined as global variables
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0

# Function for bilateral filtering.
def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image,d, 50, 50)
    return image

# Function for median filtering.
def medianFliter(image, d):
    image = cv2.medianBlur(image, d):
    return image

# Function for Inverted binary threshold.
def threshold(image, d):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, d, 255, cv2.THRESH_BINARY_INV)
    return image

# Function for dilation of objects in the image.
def dilate(image, Ksize):
    kernel = np.ones(Ksize, np.unit8)
    image = cv2.dilate(image, kernel, iterations = 1)
    return image

# Function for erosion of objects in the image.
def erode(image, Ksize):
    kernel = np.ones(Ksize, np.unit8)
    image = cv2.erode(image, kernel, iterations=1)
    return image

# Function for finding contours and straightening them horizontally. 
# Straightened lines will give better result with horizontal projections. 
def straighten(image):
    global BASELINE_ANGLE

    angle = 0.0
    angle_sum = 0.0
    contour_count = 0.0

    positive_angle_sum = 0.0
    negative_angle_sum = 0.0
    positive_count = 0.0
    negative_count = 0.0

    # Apply bilateral filter
    filtered = bilateralFilter(image, 3)
    cv2.imshow('filtered',filtered)

    # Convert to grayscale and binarize the image by INVERTED binary thresholding
    thresh = threshold(filtered, 120)
    cv2.imshow('thresh', thresh)

    # Dilate the handwritten lines in image with a suitable kernel for contour operation
    dilated = dilate(thresh, (5, 100))
    cv2.imshow('dilated', dilated)

    im2, ctrs, hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        # We can be sure the contour is not a line if height > width or height is < 20 pixels.
        # Here 20 is arbitary.
        if h>w or h<20:
            continue

        # We extract the region of interest/ contour to be straightened.
        roi = image[y:y+h, x:x+w]
        #rows, cols = ctr.shape[:2]

        # If the length of the line is less than half the document width, especially for the last line, 
        # ignore because it may yeild inacurate baseline angle which subsequently affects proceeding features. 
        if w < image.shape[1]/2:
            roi = 255
            image[y:y+h, x:x+w] = roi
            continue
        
        # minAreaRect is necessary for straightening 
        rect = cv2.minAreaRect(ctr)
        center = rect[0]
        angle = rect[2]
        # print("original: "+str(i)+" "+str(angle))

        # I actually gave a thought to this but hard to remember anyway
        if angle < -45.0:
            angle += 90.0
        # print("+90"+str(i)+" "+str(angle))

        rot = cv2.getRotationMatrix2D(((x+w)/2, (y+h)/2), angle, 1)

        # extract = cv2.warpAffine(roi, rot, (w,h), borderMode=cv2.BORDER_TRANSPARENT)
        extract = cv2.warpAffine(roi, rot, (w,h), borderMode = cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        # Image is overwritten with the straightened contour 
        image[y:y+h, x:x+w] = extract

        print(angle)
        angle_sum += angle
        contour_count += 1

        # mean angle of the contours (not lines) is found 
        mean_angle = angle_sum / contour_count
        BASELINE_ANGLE = mean_angle
        print("Average baseline angle: "+str(mean_angle))
        return image

# Function to calculate horizontal projection of the image pixel rows and return it.
def horizontalProjection(img):
    # Return a list containing the sum of pixels in each row
    (h,w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w]
        sumRows.append(np.sum(row))
    return sumRows

# Function to claculate vertical projection of the image pixel columns and return it.
def verticalProjection(img):
    # Return a list containing the sum of the pixels in each column
    (h,w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1]
        sumCols.append(np.sum(col))
    return sumCols

# Function to extract lines of handwritten text from the image using horizontal projection.
def extractLines(img):

    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN

    # Apply bilateral filter
    filtered = bilateralFilter(img, 5)

    # Convert to grayscale and bianrize the image by INVERTED binary thresholding it's better
    # to clear unwanted dark areas at the document left edge and use a high threshold value to 
    # preserve more text pixels.
    thresh = threshold(filtered, 160)
    cv2.imshow('thresh', thresh)

    # Extract a python list containing values of the horizontla projection of the image into 'hp'
    hpList = horizontalProjection(thresh)

    # Extracting 'Top Margin' feature.
    topMarginCount = 0
    for sum in hpList:
        # Sum can be strictly 0 as well. Anyway we take 0 and 255.
        if(sum<=255):
            topMarginCount +=1
        else:
            break

    # print("(Top margin row count: "+ str(topMarginCount)+")")

    # First we extract the straightened contours from the image by looking at occurance 
    # of 0's in the horizontal projection.
    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = [] # Stores the amount of space between lines.
    lines = []

    # A 2D list storing the vertical start index and end index of each contour
    # We are scanning the whole horizontal projection now
    for i, sum in enumerate(hpList):
        # Sum being 0 means blank space
        if (sum==0):
            if setSpaceTop:
                spaceTop = indexCount
                setSpaceTop = False

        # SpaceTop will be set once for each start of a space between lines
            indexCount += 1
            spaceBottom = indexCount
            if (i<len(hpList)-1):
                # This condition is necessary to avoid array index out of bound error.
                if (hpList[i+1]==0): # If the next horizontal projection is 0, keep
                    # on counting, it's still in blank space.
                    continue
                # We are using this condition if the previous contour is very thin and possibly 
                # not a line.
                if (includeNextSpace):
                    space_zero.append(spaceBottom-spaceTop)
                else:
                    if (len(space_zero)==0):
                        previous = 0

                    else:
                        previous = space_zero.pop()

                    space_zero.append(previous + spaceBottom - lineTop)
                    # Next time we encounter 0, it's begining of another space so we set new spaceTop
                    setSpaceTop = True 
                    # Sum greater than 0 means contour 
                    if(sum>0):
                        if(setLineTop):
                            lineTop = indexCount
                            setLineTop = False
                        # lineTop will be set once for each start of a new line/contour.
                        indexCount += 1
                        lineBottom  = indexCount
                        if(i<len(hpList)-1):
                            # This condition is necessary to avoid array index out of error.
                            if (hpList[i+1]>0):
                                # If the next horizontal projection is > 0, keep on counting, 
                                # it's still in contour.
                                continue

                            # If the line/contour is too thin < 10 pixels (arbitary) in height, 
                            # we ignore it.

                            # Also, we add the space following this and this contour itself to the previous 
                            # space to form a bigger space: spaceBottom - lineTop
                            if ((lineBottom-lineTop)<20):
                                includeNextSpace = False
                                setLineTop = True
                                # Next time we encounter value > 0, it's begining of another line/ contour so we 
                                # set new lineTop.
                                continue
                        
                        # The line/contours is accepted, new space following it will be accepted.
                        includeNextSpace = True 
                        
                        # Append the top and bottom horizontal indices of the line/contour in 'lines'
                        lines.append([lineTop, lineBottom])
                        setLineTop = True

                        # Next time we encounter value > 0, it's begining of another line/ contour so 
                        # we set new lineTop

    # Second we extract the very individual lines from the lines/contours we extracted above.
    fineLines = [] # A 2D list storing the horizontal start index and end index of each individual line.
    for i, line in enumerate(lines):
        anchor = line[0] # anchor will locate the horizontal indices where horizontal projection 
                            # is > ANCHOR_POINT for uphill or < ANCHOR_POINT for downhill
                            # (ANCHOR_POINT is arbitary yet suitable)
        anchorPoints = []

        # Python list where the indices obtained by 'anchor' will be stored.
        upHill = True # It implies that we expect to find the start of an individual line (vertically),
                        # climbing up the histogram

        downHill = False # It implies that we expect to find the end of an individual line (vertically),
                            # climbing down the histogram.
        
        segment = hpList[line[0]:line[1]] # We put the region of interest of the horizontal projection
                                            # of each contour here.
        
        for j, sum in enumerate(segment):
            if(upHill):
                if(sum<ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                upHill = False
                downHill = True

            if(downHill):
                if(sum > ANCHOR_POINT):
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                downHill = False
                upHill = True

        # Print Anchor Points
        # We can ignore the contour here
        if(len(anchorPoints)<2):
            continue
        # len(anchorPoints) > 3 meaning con



                        






