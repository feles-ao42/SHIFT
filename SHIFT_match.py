import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./images/inputs/Nor_7.JPG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/inputs/Nor_8.JPG', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:70], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('./images/outputs/matches_Nor.jpg', img_matches)

plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.show()

