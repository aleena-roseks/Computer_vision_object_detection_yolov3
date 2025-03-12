import cv2
import time
import math
import os

# Load Haar Cascade for vehicle detection
haar_path = "myhaar.xml"  # Change to full path if needed
if not os.path.exists(haar_path):
    print(f"Error: Haar cascade file '{haar_path}' not found!")
    exit()

carCascade = cv2.CascadeClassifier(haar_path)

# Load video
video = cv2.VideoCapture('recordo.mkv')

# Set frame dimensions
WIDTH = 1280
HEIGHT = 720

# Function to estimate speed
def estimateSpeed(location1, location2, fps):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8  # Pixels per meter (adjust for accuracy)
    d_meters = d_pixels / ppm
    if fps == 0.0:
        fps = 18  # Default FPS if unknown
    speed = d_meters * fps * 3.6  # Convert to km/h
    return speed

# Initialize tracking variables
carLocations = {}  # Stores last seen positions of cars
speed = {}  # Stores speeds of detected vehicles
frameCounter = 0

# Open output video file
out = cv2.VideoWriter('output_haar.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (WIDTH, HEIGHT))

while True:
    start_time = time.time()
    ret, frame = video.read()
    
    if not ret:
        break  # Stop if video ends

    frameCounter += 1
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    resultImage = frame.copy()
    
    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect vehicles
    cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
    
    for i, (_x, _y, _w, _h) in enumerate(cars):
        x, y, w, h = int(_x), int(_y), int(_w), int(_h)
        
        # Check for speed estimation
        if i in carLocations:
            prev_location = carLocations[i]
            speed[i] = estimateSpeed(prev_location, [x, y, w, h], 10)  # Adjust FPS if needed
            
        # Set bounding box color
        box_color = (0, 255, 0)  # Green (normal)
        if i in speed and speed[i] > 20:
            box_color = (0, 0, 255)  # Red for overspeeding

        # Draw bounding box
        cv2.rectangle(resultImage, (x, y), (x + w, y + h), box_color, 2)
        
        # Display speed label
        if i in speed:
            speed_text = f"{int(speed[i])} km/h"
            cv2.putText(resultImage, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            # Display "OVERSPEED" if speed > 20
            if speed[i] > 20:
                cv2.putText(resultImage, "OVERSPEED", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        # Update location
        carLocations[i] = [x, y, w, h]

    # Display frame
    cv2.imshow('Vehicle Detection', resultImage)
    
    # Write to output video
    out.write(resultImage)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video.release()
out.release()
cv2.destroyAllWindows()
