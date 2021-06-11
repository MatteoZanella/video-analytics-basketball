import cv2
import detection

# Define the detection model
yolo_model = detection.yolo_model()


p0 = []
rects = []
prev_window = None

frame_idx = 0
cap = cv2.VideoCapture("CV_basket.mp4")
while cap.isOpened():
    # Read one frame
    ret, frame = cap.read()
    if not ret:
        break
    # Cut the frame to a smaller window
    window = frame.copy()
    window[:] = 0
    window[500:750, 100:1300] = frame[500:750, 100:1300]
    # window = window[0:1200, 100:1300]
    window = window[500:750, 100:1300]
    window = cv2.resize(window, (window.shape[1]*2, window.shape[0]*2))
    grey = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

    # ELABORATIONS
    # People detection

    rects = detection.hog(hog, window)
    # Person tracking
    # color = np.random.randint(0, 255, (100, 3))
    # if frame_idx < 1:
    #     p0 = cv2.goodFeaturesToTrack(grey, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)
    # else:
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(prev_window, window, p0, None)
    #     good_new = p1[st == 1]
    #     good_old = p0[st == 1]
    #     for i, (new, old) in enumerate(zip(good_new, good_old)):
    #         a, b = new.ravel()
    #         c, d = old.ravel()
    #         window = cv2.circle(window, (int(a), int(b)), 5, color[i].tolist(), -1)
    # Change in ball possession detection
    # Final chores
    prev_window = window

    # PLOTTING INFORMATION
    for rect in rects:
        # rect format: (x, y, w, h)
        cv2.rectangle(window, rect, (0, 255, 0), 2)
    cv2.rectangle(window, (0, 0), (64, 128), (255.0, 0, 255.0), thickness=3)
    # Re-insert the window in the frame
    # Show results
    cv2.imshow("Working area", window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
