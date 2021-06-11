import cv2


def yolo_model():
    model = cv2.dnn.readNet("YOLO/yolov3-spp.weights", "YOLO/yolov3-spp.cfg")
    model = cv2.dnn_DetectionModel(model)
    model.setInputParams(size=(640, 640), scale=1 / 255, swapRB=True, crop=False)


def yolo(model, frame, threshold=0.3, nms_threshold=0.5):
    classes, _, rects = model.detect(frame, threshold, nms_threshold)
    people_rects = [box for (class_id, box) in zip(classes, rects) if class_id.item() == 0 and box[2] < 100]
    return people_rects


def haar_model():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')


def haar(model, frame):
    # Greyscale frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    # People detector parameters: (scaleFactor, minNeighbors, flags, minSize, maxSize)
    people_rects = model.detectMultiScale(grey, 1.3, 5)
    return people_rects


def hog_model():
    model = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return model


def hog(model, frame):
    # Greyscale frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    people_rects, _ = model.detectMultiScale(grey, winStride=(4, 4), scale=1.01)
    return people_rects
