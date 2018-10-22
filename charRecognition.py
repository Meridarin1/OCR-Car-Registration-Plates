import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tes
from PIL import Image
import cv2
import PossibleChar
import os

HEIGHT_MARGIN = 10
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

DISTANCE_BETWEEN_CHARS = 50

kNearest = cv2.ml.KNearest_create()

# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 4)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 100.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh
# end function


###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function


###################################################################################################
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function


###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:                                                                                 # if file could not be opened
        print("error, unable to open classifications.txt, exiting program\n")  # show error message
        os.system("pause")
        return False                                                                        # and return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:                                                                                 # if file could not be opened
        print("error, unable to open flattened_images.txt, exiting program\n")  # show error message
        os.system("pause")
        return False                                                                        # and return False
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(1)                                                             # set default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object

    return True                             # if we got here training was successful so return true
# end function


#######################################################################################################################
def checkInLine(poschars, img_height):
    i=0
    max = 0
    max_height = 0
    height = img_height - HEIGHT_MARGIN

    while(height > HEIGHT_MARGIN):
        i = 0
        for r in poschars:
            if(r.boundingRect[1] > height - HEIGHT_MARGIN and r.boundingRect[1] < height + HEIGHT_MARGIN):
                i = i + 1

        if(i > max):
            max = i
            max_height = height

        height -= 1
    return max_height
# end function


#######################################################################################################################
def appendPossibleRects(poschars, maxHeight):

    newPossibleRects = []

    for i in poschars:
        if(i.boundingRect[1] > maxHeight - HEIGHT_MARGIN and i.boundingRect[1] < maxHeight + HEIGHT_MARGIN):
            newPossibleRects.append(i)

    return newPossibleRects
# end function


#######################################################################################################################
def drawRects(poschars, img):

    for i in poschars:
        x1 = i.boundingRect[0]
        y1 = i.boundingRect[1]
        x2 = i.boundingRect[0] + i.boundingRect[2]
        y2 = i.boundingRect[1] + i.boundingRect[3]

        cv2.rectangle(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
# end function


#######################################################################################################################
def cropImageList(poschars, img):

    cropedImage = []

    for p in poschars:
        cropedImage.append(cropImage(p, img))

    return cropedImage
# end function


#######################################################################################################################
def cropImage(poschar, img):

    x1 = poschar.boundingRect[0]
    y1 = poschar.boundingRect[1]
    x2 = poschar.boundingRect[0] + poschar.boundingRect[2]
    y2 = poschar.boundingRect[1] + poschar.boundingRect[3]

    croped = img[y1:y2, x1:x2]
    croped = cv2.resize(croped, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
    return croped
# end function


#######################################################################################################################
def checkIfContourIsCharacter(possibleChars, contours):
    for i in contours:
        poschar = PossibleChar.PossibleChar(i)

        if(poschar.checkIfPossibleChar()):
            possibleChars.append(poschar)
# end function


#######################################################################################################################
def removeOverlaped(possibleChars):

    overlaped = []

    for pos in possibleChars:
        for p in possibleChars:
            if pos != p:
                if pos.overlap(p):
                    overlaped.append(p)
    for o in overlaped:
        possibleChars.remove(o)

    return possibleChars
# end function


#######################################################################################################################
def recognizeChars(cropedImage):
    imgROIResized = cv2.resize(cropedImage, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
    npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
    npaROIResized = np.float32(npaROIResized)

    retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)  # finally we can call findNearest !!!

    strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

    return strCurrentChar
# end function


#######################################################################################################################
def sortImage(poschar):
    for i in range(len(poschar)):
        for j in range(len(poschar)-1):
            if poschar[j].boundingRect[0] > poschar[j+1].boundingRect[0]:
                temp = poschar[j]
                poschar[j] = poschar[j+1]
                poschar[j+1] = temp
# end function


#######################################################################################################################
def connectImages(images):

    width = 0
    height = 0

    for i in images:
        h, w = i.shape
        width += w
        if h > height:
            height = h
    space_width = 10
    width += 300
    height += 100

    blank_image = np.ones((height, width), np.uint8) * 255
    space = np.ones((RESIZED_CHAR_IMAGE_HEIGHT, space_width), np.uint8) * 255

    for i, val in enumerate(images):
        # if i == 0:
        #     blank_image[50:50 + RESIZED_CHAR_IMAGE_HEIGHT, 50:50 + RESIZED_CHAR_IMAGE_WIDTH] = val
        # else:
        #     blank_image[50:50 + RESIZED_CHAR_IMAGE_HEIGHT, 50 + i * RESIZED_CHAR_IMAGE_WIDTH:50 + (i+1) * RESIZED_CHAR_IMAGE_WIDTH] = val
        if i == 0:
            blank_image[50:50 + RESIZED_CHAR_IMAGE_HEIGHT, 50:50 + RESIZED_CHAR_IMAGE_WIDTH] = val
            blank_image[50:50 + RESIZED_CHAR_IMAGE_HEIGHT, 50 + RESIZED_CHAR_IMAGE_WIDTH:50 + RESIZED_CHAR_IMAGE_WIDTH + space_width] = space
        else:
            blank_image[50:50 + RESIZED_CHAR_IMAGE_HEIGHT, 50 + i * RESIZED_CHAR_IMAGE_WIDTH + space_width * i:50 +
                        (i + 1) * RESIZED_CHAR_IMAGE_WIDTH + space_width * i] = val
    cv2.imshow('connected', blank_image)
    return blank_image
# end function


#######################################################################################################################
def checkIfCharIsNear(poschars, char):

    for p in poschars:
        distance = char.boundingRect[0] - p.boundingRect[0]

        if distance < DISTANCE_BETWEEN_CHARS and char != p:
            if distance > -DISTANCE_BETWEEN_CHARS:
                return True

    return False
# end function


#######################################################################################################################
def deleteNearChar(finalCharList):
    near = []
    for f in finalCharList:
        if not checkIfCharIsNear(finalCharList, f):
            near.append(f)

    for n in near:
        finalCharList.remove(n)

    return finalCharList
# end function


lower_range = np.array([169, 0, 220], dtype=np.uint8)
upper_range = np.array([189, 55, 255], dtype=np.uint8)

#######################################################################################################################
def main():
    # Read image with converting ot grayscale plus thresholding
    img = cv2.imread("bazarejestracji/104.jpg")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_range, upper_range)

    output = cv2.bitwise_and(hsv, hsv, mask=mask)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret3, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    cv2.imshow('THRESH', thresh)

    # gray, thresh = preprocess(img)
#    loadKNNDataAndTrainKNN()
    # Get image dimension
    height, width, channels = img.shape
    # finding all contours in the image
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    possibleChars = []
    cropedImages = []

    # Checking if contour is a character
    checkIfContourIsCharacter(possibleChars, contours)

    # Remove overlaped ones
    possibleChars = removeOverlaped(possibleChars)



    # Return Rects that are in line
    finalCharList = appendPossibleRects(possibleChars, checkInLine(possibleChars, height))
    # finalCharList = possibleChars

    finalCharList = deleteNearChar(finalCharList)



    sortImage(finalCharList)
    # Printing on image contours
    for i in possibleChars:
        cv2.drawContours(img, i.contour, 0, [0, 0, 255])

    # Crop Images
    cropedImages = cropImageList(finalCharList, gray)

    conImage = connectImages(cropedImages)

    # for i, c in enumerate(cropedImages):
    #     cropedImages[i] = cv2.cvtColor(cropedImages[i], cv2.COLOR_BGR2GRAY)

    PILImage = Image.fromarray(conImage)

    text = tes.image_to_string(PILImage, lang='pol')

    print(text)

    # for i, c in enumerate(cropedImages):
    #     cv2.imwrite(str(i) + ".tif", c)

    cv2.imwrite("pol.arklas.exp18.tif", conImage)

    # Draw Rects
    drawRects(finalCharList, img)

    # Resizing for better visualization


    # Printing img with chars bounding rects
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# end main


main()

