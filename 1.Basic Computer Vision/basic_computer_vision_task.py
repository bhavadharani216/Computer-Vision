import cv2
import os

# Get current script folder path
base_path = os.path.dirname(os.path.abspath(__file__))

# Full path of image
image_path = os.path.join(base_path, "image.jpg")

# Load image
img = cv2.imread(image_path)

# Save output in same folder
output_path = os.path.join(base_path, "output_image.jpg")
cv2.imwrite(output_path, img)

# Display image
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Output saved in same folder successfully!")
