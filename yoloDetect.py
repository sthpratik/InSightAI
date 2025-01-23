import cv2 
import numpy as np 
 
def crop_video_to_person(input_video_path, output_video_path, target_width, target_height): 
    # Load the pre-trained model (using MobileNet SSD for simplicity) 
    net = cv2.dnn.readNetFromCaffe( 
        "deploy.prototxt",  # Path to .prototxt file 
        "res10_300x300_ssd_iter_140000_fp16.caffemodel"  # Path to .caffemodel file 
    ) 
 
    # Open the video file 
    cap = cv2.VideoCapture(input_video_path) 
    if not cap.isOpened(): 
        print("Error: Unable to open the video file.") 
        return 
 
    # Get video properties 
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
 
    # Output video writer 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height)) 
 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if not ret: 
            break 
 
        # Prepare the frame for detection 
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) 
        net.setInput(blob) 
        detections = net.forward() 
 
        # Loop over detections and find the largest bounding box for a person 
        h, w = frame.shape[:2] 
        max_confidence = 0 
        best_box = None 
 
        for i in range(detections.shape[2]): 
            confidence = detections[0, 0, i, 2] 
            if confidence > 0.5:  # Confidence threshold 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
                (x1, y1, x2, y2) = box.astype("int") 
                if confidence > max_confidence: 
                    max_confidence = confidence 
                    best_box = (x1, y1, x2, y2) 
 
        if best_box is not None: 
            x1, y1, x2, y2 = best_box 
 
            # Calculate the center of the bounding box 
            center_x = (x1 + x2) // 2 
            center_y = (y1 + y2) // 2 
 
            # Ensure the crop matches the target width and height while keeping focus on the person 
            x1_crop = max(0, center_x - target_width // 2) 
            y1_crop = max(0, center_y - target_height // 2) 
            x2_crop = x1_crop + target_width 
            y2_crop = y1_crop + target_height 
 
            # Adjust if the crop goes out of frame boundaries 
            if x2_crop > w: 
                x1_crop -= (x2_crop - w) 
                x2_crop = w 
            if y2_crop > h: 
                y1_crop -= (y2_crop - h) 
                y2_crop = h 
            if x1_crop < 0: 
                x1_crop = 0 
                x2_crop = target_width 
            if y1_crop < 0: 
                y1_crop = 0 
                y2_crop = target_height 
 
            # Extract the crop 
            cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop] 
        else: 
            # If no person detected, center crop the frame 
            x1_crop = max(0, (w - target_width) // 2) 
            y1_crop = max(0, (h - target_height) // 2) 
            x2_crop = x1_crop + target_width 
            y2_crop = y1_crop + target_height 
            cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop] 
 
        # Write the frame to the output video 
        out.write(cropped_frame) 
 
    cap.release() 
    out.release() 
    print(f"Processed video saved to {output_video_path}") 
 
# Example usage 
input_video = "input.mp4" 
output_video = "output_cropped.mp4" 
target_width = 640 
target_height = 360 
video_source = 0 
crop_video_to_person(input_video, output_video, target_width, target_height) 