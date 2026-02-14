import cv2

img = cv2.imread("input.jpg")

# Resize for better visibility
img = cv2.resize(img, (800, 500))

# Line
line_img = img.copy()
cv2.line(line_img, (100, 100), (700, 100), (255, 0, 0), 5)
cv2.imwrite("output_line.jpg", line_img)

# Rectangle
rect_img = img.copy()
cv2.rectangle(rect_img, (150, 150), (650, 350), (0, 255, 0), 5)
cv2.imwrite("output_rectangle.jpg", rect_img)

# Circle
circle_img = img.copy()
cv2.circle(circle_img, (400, 250), 100, (0, 0, 255), 5)
cv2.imwrite("output_circle.jpg", circle_img)

# Ellipse
ellipse_img = img.copy()
cv2.ellipse(ellipse_img, (400, 250), (200, 100), 0, 0, 360, (255, 255, 0), 5)
cv2.imwrite("output_ellipse.jpg", ellipse_img)

# Text
text_img = img.copy()
cv2.putText(text_img, "Image Annotation", (150, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255, 0, 255), 3)
cv2.imwrite("output_text.jpg", text_img)

print("All outputs saved successfully!")
