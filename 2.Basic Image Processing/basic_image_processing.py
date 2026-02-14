import cv2
import numpy as np
import os

# Get current script folder path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load image
image_path = os.path.join(base_path, "input.jpg")
image = cv2.imread(image_path)

if image is None:
    print("Error: input.jpg not found!")
    exit()

# Save original
cv2.imwrite(os.path.join(base_path, "1_original.jpg"), image)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(base_path, "2_grayscale.jpg"), gray)

# Crop
cropped = gray[50:300, 50:300]
cv2.imwrite(os.path.join(base_path, "3_cropped.jpg"), cropped)

# Resize
resized = cv2.resize(cropped, (200, 200))
cv2.imwrite(os.path.join(base_path, "4_resized.jpg"), resized)

# Threshold
_, thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(base_path, "5_threshold.jpg"), thresh)

# Contours
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

contour_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(base_path, "6_contours.jpg"), contour_image)

# Blob detection
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(thresh)

blob_image = cv2.drawKeypoints(
    resized,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(os.path.join(base_path, "7_blob_detection.jpg"), blob_image)

print("✅ All output images saved in the same folder successfully!")
