import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込み
img1 = cv2.imread('./images/inputs/RS_14.JPG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/inputs/RS_15.JPG', cv2.IMREAD_GRAYSCALE)

# SIFT検出器を作成
sift = cv2.SIFT_create()

# 特徴点と記述子を検出
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# BFMatcherを用いて特徴点をマッチング
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# マッチング結果をソート
matches = sorted(matches, key=lambda x: x.distance)

# マッチング結果を描画
# 太さを指定して線を描画
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:100], None,
                              matchColor=(0, 255, 0), singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 線を太くして端を丸くするために再度描画
for match in matches[:100]:
    pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
    pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
    cv2.line(img_matches, pt1, pt2, (0, 255, 0), 5)  # 線の太さを3に設定

# 結果をJPG形式で保存
cv2.imwrite('./images/outputs/matches_RS_14-15.jpg', img_matches)

# 結果を表示
plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.show()
