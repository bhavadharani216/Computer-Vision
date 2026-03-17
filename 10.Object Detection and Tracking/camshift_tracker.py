import cv2

def initialize_tracker(frame):
    roi = cv2.selectROI("Select Object", frame, False)
    x, y, w, h = roi
    track_window = (x, y, w, h)

    roi_img = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return track_window, roi_hist


def track_object(frame, roi_hist, track_window):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    return ret, track_window