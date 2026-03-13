import cv2
import numpy as np

# Read input image
img = cv2.imread("6.Image segmentation/input.jpg")

# Check if image is loaded
if img is None:
    print("Error: input.jpg not found")
    exit()

# Resize image (optional)
img = cv2.resize(img, (500,350))

# Create mask
mask = np.zeros(img.shape[:2], np.uint8)

# Background and foreground models
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Define rectangle around object
rect = (50,50,400,250)

# Apply GrabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Modify mask
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Extract foreground
result = img * mask2[:,:,np.newaxis]

# Save output
cv2.imwrite("6.Image segmentation/segmented_output.png", result)

# Show images
cv2.imshow("Original Image", img)
cv2.imshow("Segmented Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Segmentation completed successfully")