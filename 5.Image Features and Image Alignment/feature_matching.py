import cv2
import numpy as np

# Read images
img1 = cv2.imread("5.Image Features and Image Alignment/image1.jpg")
img2 = cv2.imread("5.Image Features and Image Alignment/image2.jpg")
src = cv2.imread("5.Image Features and Image Alignment/object.jpg")

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Fourier Transform
# -----------------------------
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude = 20*np.log(np.abs(fshift)+1)

magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("5.Image Features and Image Alignment/fourier_output.png", magnitude)

# -----------------------------
# 2. Hough Transform (Line Detection)
# -----------------------------
edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=10)

hough_img = img1.copy()

if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(hough_img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite("5.Image Features and Image Alignment/hough_output.png", hough_img)
# -----------------------------
# 3. ORB Feature Extraction
# -----------------------------
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)

orb_img = cv2.drawKeypoints(img1,kp1,None,color=(0,255,0))

cv2.imwrite("5.Image Features and Image Alignment/orb_output.png", orb_img)
# -----------------------------
# 4. Feature Matching
# -----------------------------
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

matches = sorted(matches,key=lambda x:x.distance)

match_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None,flags=2)

cv2.imwrite("5.Image Features and Image Alignment/match_output.png", match_img)

# -----------------------------
# 5. Image Cloning
# -----------------------------
src = cv2.imread("5.Image Features and Image Alignment/object.jpg")

# Resize object image so it fits
src = cv2.resize(src, (150,150))
dst = img1.copy()

mask = 255 * np.ones(src.shape, src.dtype)
center = (dst.shape[1]//2, dst.shape[0]//2)

clone = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

cv2.imwrite("5.Image Features and Image Alignment/clone_output.png", clone)
# -----------------------------
# 6. Feature Matching Based Image Alignment
# -----------------------------
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

cv2.imwrite("5.Image Features and Image Alignment/aligned_output.png", aligned)

print("All outputs saved successfully in 5.Image Features and Image Alignment folder")