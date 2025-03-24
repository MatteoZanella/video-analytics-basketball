import cv2
import numpy as np

import detection
import tracking

# Define the detection model
detector = detection.YoloDetector("YOLO/yolov3-spp-csresnext50.weights", "YOLO/yolov3-spp-csresnext50.cfg")

# Define the tracker
tracker = tracking.CsrtTracker()

# Define the ball possession change variables
curr_side = None
possessions_changes_count = 0

# Window reduction parameters
win_rect = np.array([100, 500, 1200, 250])
win_offset = np.array([win_rect[0], win_rect[1], 0, 0])

# Video streams
cap = cv2.VideoCapture("CV_basket.mp4")
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1422, 1080))
frame_idx = 0
while cap.isOpened():
    # Read one frame
    ret, frame = cap.read()
    if not ret:
        print('Stream ended')
        break
    # Cut the frame to a smaller window
    window = frame[win_rect[1]:win_rect[1] + win_rect[3], win_rect[0]:win_rect[0] + win_rect[2]].copy()
    # CV OPERATIONS
    # 1. People detection
    detects = detector.detect(window)
    # 2. Person tracking
    if frame_idx == 0:
        # In the first frame, choose one of the detected person to be tracked
        tracker.init(window, detects[0])
    else:
        # Update the position of the tracked rectangle using the detections or the tracker
        tracker.track(window, detects)
    # 3. Possession changes detection
    centers = tracking.center(detects)
    # Only the horizontal axis is relevant for this detection
    mean_offset = np.mean(centers, axis=0)[0] - window.shape[1] / 2
    std = np.std(centers, axis=0)[0]
    if curr_side != 'left' and mean_offset < -200 and std < 200:
        possessions_changes_count += 1
        curr_side = 'left'
    if curr_side != 'right' and mean_offset > 200 and std < 200:
        possessions_changes_count += 1
        curr_side = 'right'
    # ANNOTATE INFORMATION
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (0, 255, 255)
    # 1. Detected people
    for rect in detects + win_offset:
        cv2.rectangle(frame, rect, (0, 0, 255), 1)
    cv2.putText(frame, f"N. of people detected: {len(detects)}", (20, 60), font, fontScale, color, 1, cv2.LINE_AA)
    # 2. Tracked person
    cv2.circle(frame, tracking.center(tracker.tracked) + win_offset[:2], 0, (255, 0, 255), 5, cv2.LINE_AA)
    cv2.polylines(frame, [tracker.trajectory + win_offset[:2]], False, (255, 0, 255), 1, cv2.LINE_AA)
    # 3. Ball possession changes
    cv2.putText(frame, f"N. of ball possession changes: {possessions_changes_count}", (20, 100), font, fontScale, color,
                1, cv2.LINE_AA)
    # Show the results
    cv2.imshow("Output video", frame)
    # Write the output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
