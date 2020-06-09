import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2


# show one image
def displayOne(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


# show two images
def display(a, b, title1="Original", title2="Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


def saveImg(img, path, type):
    cv2.imwrite(str(path) + "." + str(type), img)


# preprocessing
def resize(img, height=440, show=False, verbose=False):

    height = height
    width = int(height * ((img.shape[1]) / (img.shape[0])))

    dim = (width, height)
    res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if verbose:
        print('Original size', img.shape)
        print("RESIZED", res.shape)
    if show:
        displayOne(res)

    return res


def noiseReduce(img, show=False):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    if show:
        display(img, blur, 'Original', 'Blurred')
    return blur


def segmentation(img, show=False):
    img = img
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.bitwise_not(grey)
    ret, threshed = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if show:
        display(img, threshed, 'Original', 'Segmented')
    return threshed


def findEdges(img):
    image = img
    print(image.shape)
    ori = image.copy()
    image = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(gray, 75, 200)
    print("STEP 1: Edge Detection")
    plt.imshow(edged)
    plt.show()
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts[0], key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        ### Approximating the contour
        # Calculates a contour perimeter or a curve length
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        screenCnt = approx
        if len(approx) == 4:
            screenCnt = approx
            break
        # show the contour (outline)
        print("STEP 2: Finding Boundary")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    image_e = cv2.resize(image, (image.shape[1], image.shape[0]))
    cv2.imwrite('image_edge.jpg', image_e)
    plt.imshow(image_e)
    plt.show()


# segmentation(image_file)
def thresh(data, show=False):
    img = data
    # img = processing(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)

    ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    th1 = cv2.cvtColor(th1, cv2.COLOR_GRAY2RGB)
    th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
    th3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2RGB)

    titles = ['1: Original Image', '2: Global Threshold (v = 100)',
              '3: Adaptive Mean Threshold', '4: Adaptive Gaussian Threshold: 4']
    images = [img, th1, th2, th3]

    if show:
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    return images


def erode(data, erosion=True, dilation=False, show=False):
    img = cv2.imread(data, 0)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    if show:
        cv2.imshow('Input', img)
        cv2.imshow('Erosion', img_erosion)
        cv2.imshow('Dilation', img_dilation)
        cv2.waitKey(0)

    if dilation:
        return img_dilation

    return img_erosion


def detectLines(image_file):
    # does not work yet
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)
    minLineLength = 10
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('houghlines5.jpg', img)


def rotateAndScale(img, scaleFactor, degreesCCW):
    image = img
    angle = degreesCCW
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # choose a new image size.
    newX, newY = w * scaleFactor, h * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - w) / 2, (newY - h) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)), borderMode=cv2.BORDER_REPLICATE)
    return rotatedImg


def deSkew(img, show=False):
    image = img
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.bitwise_not(grey)

    # thresh image all foreground to 255 and all background to 0
    threshed = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # rotated bounding box
    coords = np.column_stack(np.where(threshed > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    rotated = rotateAndScale(image, 1, angle)

    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    if show:
        print("[INFO] angle: {:.3f}".format(angle))
        cv2.imshow("Input", image)
        cv2.imshow("Rotated", rotated)
        cv2.waitKey(0)

    return rotated


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset

    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    print(pts.size)

    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)

    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def borderTransform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = cv2.cvtColor(resize(image, height=800), cv2.COLOR_RGB2GRAY)

    img = cv2.bilateralFilter(img, 9, 75, 75)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

    img = cv2.medianBlur(img, 11)

    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    edges = cv2.Canny(img, 200, 250)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height = edges.shape[0]
    width = edges.shape[1]

    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    maxAreaFound = MAX_COUNTOUR_AREA * 0.5

    pageContour = np.array([[5, 5], [5, height - 5], [width - 5, height - 5], [width - 5, 5]])

    for cnt in contours:  # Simplify contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound
        if len(approx) == 4 and cv2.isContourConvex(approx) and maxAreaFound < cv2.contourArea(
                approx) < MAX_COUNTOUR_AREA:
            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx

    # Sort and offset corners

    pageContour = fourCornersSort(pageContour[:, 0])
    pageContour = contourOffset(pageContour, (-5, -5))

    # Recalculate to original scale - start Points

    sPoints = pageContour.dot(image.shape[0] / 800)

    # Using Euclidean distance
    # Calculate maximum height (maximal length of vertical edges) and width
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                 np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                np.linalg.norm(sPoints[3] - sPoints[0]))

    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)

    # Warping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    newImage = cv2.warpPerspective(image, M, (int(width), int(height)))

    # Convert colors back to BGR)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
    return newImage
