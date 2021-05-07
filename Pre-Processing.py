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
        # len(anchorPoints) > 3 meaning contour composed of multiple lines.
        lineTop = line[0]
        for x in range(1, len(anchorPoints)-1, 2):
            # lineMid is the horizontal index where the segmentation will be done.
            lineMid = (anchorPoints[x]+anchorPoints[x+1])/2
            lineBottom = lineMid

            # Line having height of pixels < 20 is considered defects, so we just ignore it
            # This is a weakness of the algorithm to extract lines 
            # (anchor value is ANCHOR_POINT, see for different values!)
            if(lineBottom-lineTop < 20):
                continue
            fineLines.append([lineTop, lineBottom])
            lineTop =  lineBottom
        if(line[1]-lineTop < 20):
            continue
        fineLines.append([lineTop, line[1]])

    # LINE SPACING and LETTER SIZE will be etracted here 
    # We will count the total number of pixel rows containing upper, 
    # and lower zones of the lines and add the space_zero/runs of 0's 
    # (excluding first and last of the list) to it.
    # We will count the total number of pixel rows containing midzones 
    # of the lines for letter size.

    # For this, we set an arbitary (yet suitable!) threshold MIDZONE_THRESHOLD = 15000 
    # in horizontal projection to identify the midzone conatining rows.
    
    # These two  total numbers will be divided by umber of lines 
    # (having at least one row > MIDZONE_THRESHOLD) to find average 
    # line spacing and average letter size.
    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    flag = False
    for i, line in enumerate(fineLines):
        segment = hpList[line[0]:line[1]]
        for j, sum in enumerate(segment):
            if(sum<MIDZONE_THRESHOLD):
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True

        # This line has contributed at least one count of pixel row of midzone.
        if(flag):
            lines_having_midzone_count += 1
            flag = False

    # Error Prevention -_-
    if (lines_having_midzone_count == 0): 
        lines_having_midzone_count = 1

    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1])
    # Excluding first and last entries: Top and Bottom margins
    # The number of spaces is 1 less than number of lines ut total_space_row_count
    # contains the top and bottom spaces of the line.
    average_line_spacing = float(total_space_row_count)/lines_having_midzone_count
    average_letter_size = float(midzone_row_count)/lines_having_midzone_count

    # Letter size is actually height of the letter and we are not considering width
    LETTER_SIZE = average_letter_size
    # Error prevention -_-
    if(average_letter_size == 0): average_letter_size = 1
    # We can't just take the average_line_spacing as a feature directly.
    # We must take the average_line_spacing relative to average_letter_size.
    
    # Let's take the ratio of average_line_spacing to average_letter_size as the 
    # LINE SPACING, which is perspective to average_letter_size.
    relative_line_spacing = average_line_spacing/average_letter_size
    LINE_SPACING = relative_line_spacing

    # Top margin is also taken relative to average letter size of the handwrtting
    relative_top_margin =  float(topMarginCount)/average_letter_size
    TOP_MARGIN = relative_top_margin
    
    # Showing the final extracted lines.
    for i, line in enumerate(fineLines):
        cv2.imshow("line "+str(i), img[line[0]:line[1], : ])

    print ("Average letter size: " + str(average_letter_size))
    print("Top margin relative to average letter Size: "+ 
    str(relative_top_margin))
    print("Average line spacing ralative to average letter size: "+ str(relative_line_spacing))

    return fineLines

# Function to extract words from the lines using vertical projection.
def extractWords(image, lines):
    global LETTER_SIZE 
    global WORD_SPACING

    # Apply bilateral filter
    filtered = bilateralFilter(image, 5)

    # Convert to grayscale and binarize the image by INVERTED binary thresholding 
    thresh = threshold(filtered, 180)
    # cv2.imshow('thresh', thresh)

    # Width of the whole document is found once.
    width = thresh.shape[1]
    space_zero = []
    words = []

    # Isolated words or components will be extracted fro each line by looking at occurance of 0's
    # in its vertical projection.
    for i, line in enumerate(lines):
        extract = thresh[line[0]:line[1], 0:width]
        vp = verticalProjection(extract)

        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []

        # We are scanning the vertical projection.
        for j, sum in enumerate(vp):
            # Sum being 0 means blank space.
            if (sum==0):
                if (setSpaceStart):
                    spaceStart = indexCount
                    setSpaceStart = False

                # spaceStart will be set once for each start of a space between lines.
                indexCount += 1
                spaceEnd = indexCount
                if(j< len(vp)-1):
                    # This condition is necessary to avoid array index out of bound error.
                    if (vp[j+1]==0):
                        # If the next vertical projection is 0, keep on counting, 
                        # it's still in blank space.
                        continue
                    
                # We ignore spaces which is smaller than half the average letter size.
                if ((spaceEnd-spaceStart)>int(LETTER_SIZE/2)):
                    spaces.append(spaceEnd - spaceStart)
                # Next time we encounter 0, it's being of another space so we set new spaceStart.
                setSpacesStart = True

            if(sum>0):
                if(setWordStart):
                    wordStart = indexCount
                    setWordStart = False

                # wordStart will be set once for each start of a new word/component.
                indexCount += 1
                wordEnd = indexCount
                if(j<len(vp)-1):
                    # This condition is necessary to avoid array index out of bound error.
                    if(vp[j+1]>0):
                        # If the next horizontal projection is > 0, keep on counting, 
                        # it's still in non-space zone.
                        continue

                # Append the coordinates of each word/component: y1, y2, x1, x2 in 'words'
                # We ignore the ones which has height smaller than half the average letter size.
                # This will remove full stops and commas as an individual component
                count = 0
                for k in range(line[1]-line[0]):
                    row = thresh[line[0]+k : line[0]+k+1, wordStart:wordEnd]
                    if(np.sum(row)):
                        count += 1
                if (count > int(LETTER_SIZE/2)):
                    words.append([line[0], line[1], wordStart, wordEnd])

                # Next time we encounter value > 0, it's begining of another word/component 
                # so we set new wordStart
                setWordStart = True
        space_zero.extend(spaces[1:-1])

    # Print space_zero
    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if(space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns)/ space_count
    relative_word_spacing = average_word_spacing/LETTER_SIZE
    WORD_SPACING = relative_word_spacing
    # print("Average word spacing: "+str(average_word_spacing))
    print("Average word spacing relative to average letter size: " + str(relative_word_spacing))

    return words






               







                        






