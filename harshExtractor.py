#!/usr/bin/python3
# 2018.01.16 01:11:49 CST
# 2018.01.16 01:55:01 CST
import cv2
import numpy as np
import math

# Features to be extracted
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
    image = cv2.medianBlur(image, d)
    return image

# Function for Inverted binary threshold.
def threshold(image, d):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, d, 255, cv2.THRESH_BINARY_INV)
    return image

# Function for dilation of objects in the image.
def dilate(image, Ksize):
    kernel = np.ones(Ksize, np.uint8)
    image = cv2.dilate(image, kernel, iterations = 1)
    return image

# Function for erosion of objects in the image.
def erode(image, Ksize):
    kernel = np.ones(Ksize, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image


# Function to calculate horizontal projection of the image pixel rows and return it.
def horizontalProjection(img):
    # Return a list containing the sum of the pixels in each row
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w] # y1:y2, x1:x2
        sumRows.append(np.sum(row))
    return sumRows
    
# Function to calculate vertical projection of the image pixel columns and return it.
def verticalProjection(img):
    # Return a list containing the sum of the pixels in each column
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1] # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols

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
    # cv2.imshow('filtered',filtered)

    # Convert to grayscale and binarize the image by INVERTED binary thresholding
    thresh = threshold(filtered, 120)
    # cv2.imshow('thresh', thresh)

    # Dilate the handwritten lines in image with a suitable kernel for contour operation
    dilated = dilate(thresh, (5, 100))
    # cv2.imshow('dilated', dilated)

    ctrs,im2 = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        # print("Average baseline angle: "+str(mean_angle))
        return image

def extractLines(img):

    global TOP_MARGIN
    global LETTER_SIZE
    global LINE_SPACING
    
    ## (1) read
    # img = cv2.imread("C:\\Users\Harsh\Desktop\Projects\Handwriting-Analysis-using-Machine-Learning\Test Images/a01-000u.png")
    # img = cv2.resize(img, (1280,720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[int(img.shape[0]/5.76):int(img.shape[0]/1.27), 0:img.shape[1]]

    ## (2) threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    ## (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    # if w>h:
    # #     w,h = w,h
    #     ang += 90

    ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx,cy), 0, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

    th = 2
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    average_letter_size = 0
    average_line_space = 0
    print(uppers[0])
    print(lowers[0])
    for i in range(0, len(uppers)):
        average_letter_size = average_letter_size + (lowers[i] - uppers[i])
        if (i<(len(uppers)-1)):
            average_line_space = average_line_space + (uppers[i+1] - lowers[i])

    average_letter_size = average_letter_size/len(uppers)
    average_line_space = average_line_space/len(uppers)
    top_margin = uppers[0]

    TOP_MARGIN = top_margin
    LETTER_SIZE = average_letter_size
    LINE_SPACING = average_line_space
    print("Top Margin", TOP_MARGIN)
    print("Letter size",LETTER_SIZE)
    print("Line Spacing", LINE_SPACING)



    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    for y in uppers:
        cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

    for y in lowers:
        cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)

    cv2.imshow("result.png", rotated)
    return [uppers,lowers]

''' function to extract words from the lines using vertical projection '''
def extractWords(image, lines):

    global LETTER_SIZE
    global WORD_SPACING
    
    # apply bilateral filter
    filtered = bilateralFilter(image, 5)
    
    # convert to grayscale and binarize the image by INVERTED binary thresholding
    thresh = threshold(filtered, 180)
    #cv2.imshow('thresh', wthresh)
    
    # Width of the whole document is found once.
    width = thresh.shape[1]
    space_zero = [] # stores the amount of space between words
    words = [] # a 2D list storing the coordinates of each word: y1, y2, x1, x2
    
    # Isolated words or components will be extacted from each line by looking at occurance of 0's in its vertical projection.
    for i, line in enumerate(lines):
        extract = thresh[line[0]:line[1], 0:width] # y1:y2, x1:x2
        vp = verticalProjection(extract)
        #print i
        #print vp
        
        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []
        
        # we are scanning the vertical projection
        for j, sum in enumerate(vp):
            # sum being 0 means blank space
            if(sum==0):
                if(setSpaceStart):
                    spaceStart = indexCount
                    setSpaceStart = False # spaceStart will be set once for each start of a space between lines
                indexCount += 1
                spaceEnd = indexCount
                if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                    if(vp[j+1]==0): # if the next vertical projectin is 0, keep on counting, it's still in blank space
                        continue

                # we ignore spaces which is smaller than half the average letter size
                if((spaceEnd-spaceStart) > int(LETTER_SIZE/2)):
                    spaces.append(spaceEnd-spaceStart)
                    
                setSpaceStart = True # next time we encounter 0, it's begining of another space so we set new spaceStart
            
            # sum greater than 0 means word/component
            if(sum>0):
                if(setWordStart):
                    wordStart = indexCount
                    setWordStart = False # wordStart will be set once for each start of a new word/component
                indexCount += 1
                wordEnd = indexCount
                if(j<len(vp)-1): # this condition is necessary to avoid array index out of bound error
                    if(vp[j+1]>0): # if the next horizontal projectin is > 0, keep on counting, it's still in non-space zone
                        continue
                
                # append the coordinates of each word/component: y1, y2, x1, x2 in 'words'
                # we ignore the ones which has height smaller than half the average letter size
                # this will remove full stops and commas as an individual component
                count = 0
                for k in range(line[1]-line[0]):
                    row = thresh[line[0]+k:line[0]+k+1, wordStart:wordEnd] # y1:y2, x1:x2
                    if(np.sum(row)):
                        count += 1
                if(count > int(LETTER_SIZE/2)):
                    words.append([line[0], line[1], wordStart, wordEnd])
                    
                setWordStart = True # next time we encounter value > 0, it's begining of another word/component so we set new wordStart
        
        space_zero.extend(spaces[1:-1])
    
    #print space_zero
    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if(space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns) / space_count
    relative_word_spacing = average_word_spacing / LETTER_SIZE
    WORD_SPACING = relative_word_spacing
    print("Average word spacing: "+str(average_word_spacing))
    print ("Average word spacing relative to average letter size: "+str(relative_word_spacing))
    
    return words

# Function to extract average pen pressure of the handwriting.
def barometer(image):

    global PEN_PRESSURE

    # it's extremely necessary to convert to grayscale first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # inverting the image pixel by pixel individually. This costs the maximum time and processing in the entire process!
    h, w = image.shape[:]
    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]
    
    #cv2.imshow('inverted', inverted)
    
    # bilateral filtering
    filtered = bilateralFilter(inverted, 3)
    
    # binary thresholding. Here we use 'threshold to zero' which is crucial for what we want.
    # If src(x,y) is lower than threshold=100, the new pixel value will be set to 0, else it will be left untouched!
    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    #cv2.imshow('thresh', thresh)
    
    # add up all the non-zero pixel values in the image and divide by the number of them to find the average pixel value in the whole image
    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(thresh[x][y] > 0):
                total_intensity += thresh[x][y]
                pixel_count += 1
                
    average_intensity = float(total_intensity) / pixel_count
    PEN_PRESSURE = average_intensity
    #print total_intensity
    #print pixel_count
    print ("Average pen pressure: "+str(average_intensity))

    return 

def start(img):

    global BASELINE_ANGLE
    global TOP_MARGIN
    global LETTER_SIZE
    global LINE_SPACING
    global WORD_SPACING
    global PEN_PRESSURE
    global SLANT_ANGLE

    img = cv2.imread(img)
    
    # Base Line angle
    straighten(img)

    # Line Extraction
    lines = extractLines(img)

    # Word Spacing
    extractWords(img, lines)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

start("C:\\Users\Harsh\Desktop\Projects\Handwriting-Analysis-using-Machine-Learning\Test Images/a01-000u.png")