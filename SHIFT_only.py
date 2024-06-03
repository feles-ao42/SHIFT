import cv2
import numpy as np

img = cv2.imread('./images/inputs/sample.JPG')
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, keypoints, None, flags=4)
cv2.imwrite("./images/outputs/sample_match.jpg",img_sift)
