import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your specific model file

# Parameters
input_source = 0  # Use 0 for webcam or provide video file path
output_video = 'output.mp4'  # Output video file
output_width = 640  # Desired output width
output_height = 480  # Desired output height

# Open video source
cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS)) if input_source != 0 else 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))

# Initialize variables to smooth flickering
prev_x_min, prev_y_min, prev_x_max, prev_y_max = 0, 0, output_width, output_height
alpha = 0.5  # Smoothing factor

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 inference
    results = model(frame)

    # Get bounding boxes
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    if len(detections) > 0:
        # Get the first detected object's bounding box
        x_min, y_min, x_max, y_max = detections[0][:4]

        # Draw bounding box on the original frame
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Smooth bounding box coordinates
        x_min = int(alpha * x_min + (1 - alpha) * prev_x_min)
        y_min = int(alpha * y_min + (1 - alpha) * prev_y_min)
        x_max = int(alpha * x_max + (1 - alpha) * prev_x_max)
        y_max = int(alpha * y_max + (1 - alpha) * prev_y_max)

        prev_x_min, prev_y_min, prev_x_max, prev_y_max = x_min, y_min, x_max, y_max

        # Ensure the entire object is visible in the crop
        obj_width = x_max - x_min
        obj_height = y_max - y_min
        crop_width = max(output_width, obj_width)
        crop_height = max(output_height, obj_height)

        # Calculate the center of the bounding box
        obj_center_x = int((x_min + x_max) / 2)
        obj_center_y = int((y_min + y_max) / 2)

        # Calculate the cropping box
        crop_x_min = max(0, obj_center_x - crop_width // 2)
        crop_y_min = max(0, obj_center_y - crop_height // 2)
        crop_x_max = crop_x_min + crop_width
        crop_y_max = crop_y_min + crop_height

        # Adjust the cropping box if it exceeds frame dimensions
        if crop_x_max > frame.shape[1]:
            crop_x_min -= crop_x_max - frame.shape[1]
            crop_x_max = frame.shape[1]
        if crop_y_max > frame.shape[0]:
            crop_y_min -= crop_y_max - frame.shape[0]
            crop_y_max = frame.shape[0]

        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)

        # Crop the frame
        cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # Resize the frame to the desired output dimensions
        resized_frame = cv2.resize(cropped_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    else:
        # No detection, resize the original frame to output dimensions
        resized_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Show both original and processed frames side by side
    combined_frame = np.hstack((cv2.resize(frame, (output_width, output_height)), resized_frame))
    cv2.imshow('Original (Left) | Processed (Right)', combined_frame)

    # Write the processed frame to the output video
    out.write(resized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as:", output_video)
