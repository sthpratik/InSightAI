import cv2
import torch
import numpy as np
from ultralytics import YOLO

def get_video_source(input_source):
    print('Reading video from website')
    if isinstance(input_source, str) and (input_source.startswith("http://") or input_source.startswith("https://")):
        return cv2.VideoCapture(input_source)
    try:
        return cv2.VideoCapture(int(input_source))
    except ValueError:
        return cv2.VideoCapture(input_source)

def detect_and_focus_people(input_source, output_dimensions=(640, 480), debug=False):
    # Load YOLOv8 model
    model = YOLO('yolov10n.pt')  # Replace 'yolov8n.pt' with your specific model file

    # Open video source
    cap = get_video_source(input_source)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        exit()

    # Get original frame dimensions
    (output_width, output_height) = output_dimensions  # Desired output width and height

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = output_width / output_height
    
    output_video = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if input_source != 0 else 30
    out = cv2.VideoWriter(output_video, fourcc, fps, output_dimensions)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLOv8 inference
        results = model(frame)

        # Get bounding boxes
        detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

        if len(detections) > 0:
            # Compute the bounding box that contains all detected objects
            x_min = int(min(d[0] for d in detections))
            y_min = int(min(d[1] for d in detections))
            x_max = int(max(d[2] for d in detections))
            y_max = int(max(d[3] for d in detections))

            # Draw bounding boxes if debug mode is enabled
            if debug:
                for box in detections:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Expand the crop region while maintaining aspect ratio and keeping the object centered
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Adjust crop dimensions to match the target aspect ratio while including object
            if target_aspect_ratio > (crop_width / crop_height):
                new_crop_height = crop_height
                new_crop_width = int(new_crop_height * target_aspect_ratio)
            else:
                new_crop_width = crop_width
                new_crop_height = int(new_crop_width / target_aspect_ratio)
            
            crop_x_min = max(0, center_x - new_crop_width // 2)
            crop_x_max = min(original_width, crop_x_min + new_crop_width)
            crop_y_min = max(0, center_y - new_crop_height // 2)
            crop_y_max = min(original_height, crop_y_min + new_crop_height)
            
            cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            resized_frame = cv2.resize(cropped_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Input', frame)
        cv2.imshow('Output', resized_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved as:", output_video)

if __name__ == "__main__":
    # VIDEO_SOURCE = "video/input.mp4"
    # detect_and_focus_people(VIDEO_SOURCE, output_width=960, output_height=540, debug=True)
    VIDEO_SOURCE = "video/sg.mp4"
    detect_and_focus_people(VIDEO_SOURCE, (320, 180), debug=True)
