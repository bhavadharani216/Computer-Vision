import cv2
import numpy as np
import os

# Load Image
img = cv2.imread("image.jpg")

if img is None:
    print("Image not found!")
    exit()

# Get current folder path
folder_path = os.getcwd()

# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(folder_path, "1_Grayscale.jpg"), gray)

# 2. HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite(os.path.join(folder_path, "2_HSV.jpg"), hsv)

# 3. Histogram Equalization
equalized = cv2.equalizeHist(gray)
cv2.imwrite(os.path.join(folder_path, "3_Histogram_Equalized.jpg"), equalized)

# 4. Sharpening (Convolution)
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])

sharpened = cv2.filter2D(img, -1, kernel)
cv2.imwrite(os.path.join(folder_path, "4_Sharpened.jpg"), sharpened)

# 5. Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5,5), 0)
cv2.imwrite(os.path.join(folder_path, "5_Gaussian_Blur.jpg"), gaussian)

# 6. Median Blur
median = cv2.medianBlur(img, 5)
cv2.imwrite(os.path.join(folder_path, "6_Median_Blur.jpg"), median)

# 7. Sobel X
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
cv2.imwrite(os.path.join(folder_path, "7_Sobel_X.jpg"), sobelx)

# 8. Sobel Y
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
cv2.imwrite(os.path.join(folder_path, "8_Sobel_Y.jpg"), sobely)

# 9. Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite(os.path.join(folder_path, "9_Canny_Edges.jpg"), edges)

print("All output images saved successfully in:", folder_path)