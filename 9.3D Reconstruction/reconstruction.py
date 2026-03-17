import cv2
import numpy as np

# Load images (grayscale)
imgL = cv2.imread(r"E:\PROJECTS\Computer-Vision\9.3D Reconstruction\left.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(r"E:\PROJECTS\Computer-Vision\9.3D Reconstruction\right.png", cv2.IMREAD_GRAYSCALE)

# Check images
if imgL is None or imgR is None:
    print("Error: Images not found. Check file path.")
    exit()

# Ensure same size
imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))

# Optional: Histogram equalization (improves matching)
imgL = cv2.equalizeHist(imgL)
imgR = cv2.equalizeHist(imgR)

# Create StereoSGBM matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,   # must be multiple of 16
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Compute disparity
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Remove invalid values
disparity[disparity < 0] = 0

# Normalize for visualization
depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
depth_map = np.uint8(depth_map)

# Apply color map (🔥 makes output look GOOD for record)
depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

# Save outputs
cv2.imwrite(r"E:\PROJECTS\Computer-Vision\9.3D Reconstruction\depth_map_gray.png", depth_map)
cv2.imwrite(r"E:\PROJECTS\Computer-Vision\9.3D Reconstruction\depth_map_color.png", depth_colored)

# Show results
cv2.imshow("Left Image", imgL)
cv2.imshow("Right Image", imgR)
cv2.imshow("Depth Map (Gray)", depth_map)
cv2.imshow("Depth Map (Color)", depth_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Depth map generated successfully")