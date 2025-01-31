import cv2
import numpy as np

# Define the desired output video size
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480

def main(video_path=None):
    print(cv2.__version__)
    # Open video file or webcam
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video frame")
        cap.release()
        return

    # Select the object to track
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    # Initialize tracker
    tracker = cv2.TrackerCSRT_create()
    success = tracker.init(frame, bbox)
    if not success:
      print("Tracker failed to initialize.")
      return

    while True:
        ret, frame = cap.read()
        if not ret:
          print("Frame not read properly.")
          break

        # Update tracker
        success, bbox = tracker.update(frame)
        print(f"Tracking success: {success}, Bbox: {bbox}")

        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x, center_y = x + w // 2, y + h // 2

            # Calculate cropping region
            start_x = max(0, center_x - OUTPUT_WIDTH // 2)
            start_y = max(0, center_y - OUTPUT_HEIGHT // 2)
            end_x = min(frame.shape[1], start_x + OUTPUT_WIDTH)
            end_y = min(frame.shape[0], start_y + OUTPUT_HEIGHT)

            cropped_frame = frame[start_y:end_y, start_x:end_x]
            
            # Resize if needed (to ensure fixed output size)
            cropped_frame = cv2.resize(cropped_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        else:
            cropped_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

        # Display the output
        cv2.imshow("Tracked Object", frame)
        cv2.imshow("Cropped Output", cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("video/walgreen.mp4")
