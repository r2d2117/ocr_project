import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2
import preprocess as pp
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
ap.add_argument("-f", "--folder", required=False, help="Path to directory to scan")
args = vars(ap.parse_args())

def loadImages(path):
    image_files = sorted([os.path.join(path, 'images', file)
                          for file in os.listdir(path + "/images") if
                          file.endswith('.png')])

    return image_files

# loading image
img = cv2.imread(args["image"])
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.bilateralFilter(img, 9, 75, 75)
# img = cv2.medianBlur(img, 11)
# img = pp.resize(img, height=800)
# cv2.imshow("img", img)
# cv2.waitKey()
# cv2.destroyAllWindows()


