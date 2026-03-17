import cv2
from camshift_tracker import initialize_tracker, track_object
from kalman_filter import KalmanTracker

cap = cv2.VideoCapture("video.mp4")

ret, frame = cap.read()
if not ret:
    print("Camera error")
    exit()

# Select object
track_window, roi_hist = initialize_tracker(frame)

# Kalman filter
kalman = KalmanTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # CamShift tracking
    ret_cam, track_window = track_object(frame, roi_hist, track_window)

    x, y, w, h = track_window
    center = (int(x + w/2), int(y + h/2))

    # Kalman prediction
    pred_x, pred_y = kalman.predict(center)

    # Draw CamShift box
    pts = cv2.boxPoints(ret_cam)
    pts = pts.astype(int)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # Draw Kalman point
    cv2.circle(frame, (int(pred_x), int(pred_y)), 8, (255, 0, 0), -1)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
