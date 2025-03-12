import cv2
import time
import math
import os
import numpy as np

# Load Haar Cascade for bike detection
haar_path = "motor-v4.xml"
if not os.path.exists(haar_path):
    print(f"Error: Haar cascade file '{haar_path}' not found!")
    exit()

bike_cascade = cv2.CascadeClassifier(haar_path)

# Load YOLO for helmet detection
helmet_cfg = "yolov3-obj.cfg"  
helmet_weights = "yolov3-obj_2400.weights"  
helmet_net = cv2.dnn.readNetFromDarknet(helmet_cfg, helmet_weights)

# Load class names (should include 'Helmet')

classes = "helmet"

# Get YOLO output layers
def getOutputsNames(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect helmets in a frame
def detectHelmets(frame, threshold=0.5):  # Increased threshold from 0.3 to 0.5 for better accuracy
    blob = cv2.dnn.blobFromImage(frame, 1/255, (608, 608), [0, 0, 0], 1, crop=False)  # Increased input size
    helmet_net.setInput(blob)
    outs = helmet_net.forward(getOutputsNames(helmet_net))

    frameHeight, frameWidth = frame.shape[:2]
    helmet_boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold and classes[classId] == "Helmet":
                center_x, center_y, width, height = (
                    int(detection[0] * frameWidth),
                    int(detection[1] * frameHeight),
                    int(detection[2] * frameWidth),
                    int(detection[3] * frameHeight),
                )
                left, top = int(center_x - width / 2), int(center_y - height / 2)
                helmet_boxes.append([left, top, width, height, center_x])  # Added helmet center_x

    return helmet_boxes

# Load video
video = cv2.VideoCapture('overspeed.mp4')

# Get video properties
WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(video.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter('output_bike_helmet.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))

# Initialize tracking variables
bikeLocations = {}  # Stores last known locations of bikes
speed = {}  # Stores speed values
frameCounter = 0

# Function to estimate speed
def estimateSpeed(location1, location2, fps):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8  # Pixels per meter (adjust for accuracy)
    d_meters = d_pixels / ppm
    if fps == 0.0:
        fps = 18  # Default FPS if unknown
    speed = d_meters * fps * 3.6  # Convert to km/h
    return speed

while True:
    start_time = time.time()
    ret, frame = video.read()
    if not ret:
        break  # Stop if video ends

    frameCounter += 1
    resultImage = frame.copy()

    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bikes
    bikes = bike_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Detect helmets
    helmet_boxes = detectHelmets(frame)

    for i, (x, y, w, h) in enumerate(bikes):
        has_helmet = False

        # Estimate speed
        if i in bikeLocations:
            prev_location = bikeLocations[i]
            speed[i] = estimateSpeed(prev_location, [x, y, w, h], FPS)  # Adjust FPS if needed

        # Check if a helmet is detected above the bike and centered
        for (hx, hy, hw, hh, helmet_center_x) in helmet_boxes:
            bike_center_x = x + w // 2  # Calculate bike center
            if hy < y and abs(helmet_center_x - bike_center_x) < w // 3:  # Ensure helmet is centered
                has_helmet = True
                break

        # Set bounding box color
        if has_helmet:
            box_color = (255, 0, 0)  # Blue for bikes with helmets
        else:
            box_color = (0, 0, 255)  # Red for bikes without helmets
            cv2.putText(resultImage, "NO HELMET", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if i in speed and speed[i] > 35:
            box_color = (128, 0, 128)  # Purple for overspeeding (>35 km/h)

        # Draw bounding box
        cv2.rectangle(resultImage, (x, y), (x + w, y + h), box_color, 2)

        # Display speed label
        if i in speed:
            speed_text = f"{int(speed[i])} km/h"
            cv2.putText(resultImage, speed_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Update bike location
        bikeLocations[i] = [x, y, w, h]

    # Display frame
    cv2.imshow('Bike, Speed & Helmet Detection', resultImage)

    # Write output video
    out.write(resultImage)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video.release()
out.release()
cv2.destroyAllWindows()
