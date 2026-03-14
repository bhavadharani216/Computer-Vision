import cv2
import numpy as np
import os

# Get path of current script folder
folder = os.path.dirname(__file__)

# Read image from same folder
img = cv2.imread(os.path.join(folder, "person.jpg"))

if img is None:
    print("Error: person.jpg not found")
    exit()

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect person
boxes, weights = hog.detectMultiScale(img, winStride=(8,8))

# Draw bounding box
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# Save output in same folder
output_path = os.path.join(folder, "pose_output.png")
cv2.imwrite(output_path, img)

# Show result
cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Output saved in the same folder")