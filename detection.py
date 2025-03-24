import cv2
import numpy as np


class YoloDetector:
    def __init__(self, weights, cfg):
        model = cv2.dnn.readNet(weights, cfg)
        model = cv2.dnn_DetectionModel(model)
        model.setInputParams(size=(608, 608), scale=1 / 255, swapRB=True, crop=False)
        self.model = model

    def detect(self, img, threshold=0.35, nms_threshold=0.6):
        # Define the padding to square the images
        pad = int((img.shape[1] / 2 + 15 - img.shape[0]) / 2)
        left_img = cv2.copyMakeBorder(img[:, :int(img.shape[1] / 2) + 15], pad, pad, 0, 0, cv2.BORDER_CONSTANT)
        right_img = cv2.copyMakeBorder(img[:, int(img.shape[1] / 2) - 15:], pad, pad, 0, 0, cv2.BORDER_CONSTANT)
        # Detect the people
        left_detects = self._detect(left_img, threshold, nms_threshold)
        right_detects = self._detect(right_img, threshold, nms_threshold)
        # Align the right detections w.r.t the padded window
        right_detects[:, 0] += int(img.shape[1] / 2) - 15
        # Extract the shared detections
        left_shared = left_detects[left_detects[:, 0] + left_detects[:, 2] >= img.shape[1] / 2 - 15]
        left_detects = left_detects[~(left_detects[:, 0] + left_detects[:, 2] >= img.shape[1] / 2 - 15)]
        left_shared = left_shared[left_shared[:, 0] + left_shared[:, 2] < img.shape[1] / 2 + 13]
        right_shared = right_detects[right_detects[:, 0] <= img.shape[1] / 2 + 15]
        right_detects = right_detects[~(right_detects[:, 0] <= img.shape[1] / 2 + 15)]
        right_shared = right_shared[right_shared[:, 0] > img.shape[1] / 2 - 13]
        shared = np.vstack([left_shared, right_shared])
        # Remove the overlapping boxes
        if len(shared) > 0:
            indices = cv2.dnn.NMSBoxes(shared[:, :-1].tolist(), shared[:, -1], threshold, 0.25)
            shared = np.take(shared, indices.reshape(-1), axis=0)
        # Merge the detections together
        detects = np.vstack([left_detects, shared, right_detects])
        # Align all the detections w.r.t the initial window
        detects[:, 1] -= pad
        return detects[:, :-1].astype(int)

    def _detect(self, img, threshold, nms_threshold):
        # Detect in the image, removing non-humans and out of size detections
        classes, scores, rects = self.model.detect(img, threshold, nms_threshold)
        detections = np.array([np.append(rect, score) for (class_id, rect, score) in zip(classes, rects, scores)
                               if class_id.item() == 0 and 20 * 20 < rect[2] * rect[3] < 100 * 100])
        return detections if len(detections) > 0 else np.empty((0, 5))


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
