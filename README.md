# Basketball Tracking

This project implements a computer vision system for detecting and tracking player movements and therefore deducing the number of ball possession changes in basketball videos using advanced object detection and tracking techniques.

https://github.com/user-attachments/assets/bc479f56-c99c-4bcd-92f5-a828a663d81b

## Key Features
- **Person Detection**: Uses YOLO (You Only Look Once) deep learning object detection to find the players in the scene
- **Player Tracking**: Implements CSRT (Channel and Spatial Reliability Tracker) for tracking one of the players for the whole video duration
- **Possession Change Detection**: Analyzes player positions to determine ball possession changes
- **Video Annotation**: Provides real-time visualization of detections, tracking, and possession changes

## Detection Methods
The project implemented and compared multiple person detection techniques, but YOLO was used for its superior performance:
- YOLO Deep Learning Detector
- Haar Cascade Classifier
- HOG (Histogram of Oriented Gradients) Detector

## Installation
1. Clone this repository
  ```bash
  git clone https://github.com/MatteoZanella/video-analytics-basketball.git
  ```
2. Install the requirements
  ```bash
  cd video-analytics-basketball
  pip install -r requirements.txt
  ```
3. Download YOLO weights inside the `./YOLO` folder. The file [yolov3-model.txt](./YOLO/yolov3-model.txt) contains the links for the configuration and the weights adopted for this project

## Usage
To use the tool, simply run the `main.py` file
```bash
python main.py
```

In `main.py`, you can modify the the video input path and the set parameters
