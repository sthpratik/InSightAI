import cv2
import numpy as np

# Define the desired output video size
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480

def videoTracker(video_path=None):
    print(cv2.__version__)
    # Open video file or webcam
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Adjust this value as needed
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Value between 0 and 1
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Adjust contrast
    cap.set(cv2.CAP_PROP_SATURATION, 0.5)  # Adjust saturation

    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video frame")
        cap.release()
        return

    # Ensure frame is not empty
    if frame.size == 0:
        print("Error: Empty frame received")
        cap.release()
        return

    # Resize frame if too large (can help with tracking stability)
    frame = cv2.resize(frame, (640, 480))


    # Select the object to track
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    # Verify bbox is valid
    if bbox == (0, 0, 0, 0):
        print("Error: Invalid bounding box selection")
        cap.release()
        return

    # Create tracker - try different algorithms
    tracker_types = ['CSRT', 'KCF', 'MOSSE', 'MIL']
    tracker = None
    
    for tracker_type in tracker_types:
        try:
            if tracker_type == 'CSRT':
                tracker = cv2.legacy.TrackerCSRT.create()
            elif tracker_type == 'KCF':
                tracker = cv2.legacy.TrackerKCF.create()
            elif tracker_type == 'MOSSE':
                tracker = cv2.legacy.TrackerMOSSE.create()
            elif tracker_type == 'MIL':
                tracker = cv2.legacy.TrackerMIL.create()
            
            # Try to initialize the tracker
            success = tracker.init(frame, bbox)
            if success:
                print(f"Successfully initialized {tracker_type} tracker")
                break
            else:
                print(f"Failed to initialize {tracker_type} tracker, trying next...")
        except Exception as e:
            print(f"Error with {tracker_type} tracker: {str(e)}")
            continue

    if tracker is None or not success:
        print("Error: Could not initialize any tracker")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read properly.")
            break

        # Resize frame to maintain consistency
        frame = cv2.resize(frame, (640, 480))

        # Update tracker
        try:
            success, bbox = tracker.update(frame)
        except Exception as e:
            print(f"Tracking error: {str(e)}")
            success = False

        # Draw bounding box if tracking is successful
        if success:
            # Convert bbox coordinates to integers
            bbox = tuple(map(int, bbox))
            # Draw rectangle
            cv2.rectangle(frame, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                         (0, 255, 0), 2)
            
            # Add text to show tracking status
            cv2.putText(frame, "Tracking", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            # Add text to show tracking lost
            cv2.putText(frame, "Lost", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Tracking', frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    videoTracker("/Users/pratikshrestha/Documents/Apps/Research/AI/sample-videos/head-pose-face-detection-female.mp4")  # Use 0 for webcam