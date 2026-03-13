import cv2
import numpy as np


img = cv2.imread("7.Camera Calibration/grid.jpg")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Pattern size based on your grid
pattern_size = (12,8)

ret, centers = cv2.findCirclesGrid(
    gray,
    pattern_size,
    flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
)

if ret:
    print("Circular grid detected")
    cv2.drawChessboardCorners(img, pattern_size, centers, ret)
else:
    print("Grid NOT detected")

# Save output
cv2.imwrite("7.Camera Calibration/calibration_output.png", img)

cv2.imshow("Calibration Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()