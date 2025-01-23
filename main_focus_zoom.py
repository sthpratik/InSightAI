import cv2
import numpy as np

# Specify paths for the pre-trained model and video source
MODEL_PATH = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WEIGHTS_PATH = 'models/frozen_inference_graph.pb'
VIDEO_SOURCE = 0  # 0 for webcam, or replace with filepath or URL

def get_video_source(input_source):
    if isinstance(input_source, str) and (input_source.startswith("http://") or input_source.startswith("https://")):
        return cv2.VideoCapture(input_source)
    try:
        return cv2.VideoCapture(int(input_source))
    except ValueError:
        return cv2.VideoCapture(input_source)

def detect_and_focus_people(video_source, output_dimensions=(640, 480), debug=False):
    net = cv2.dnn_DetectionModel(WEIGHTS_PATH, MODEL_PATH)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    cap = get_video_source(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, output_dimensions)

    smoothed_bbox = None
    smoothing_factor = 0.9

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        classIds, confs, bbox = net.detect(frame, confThreshold=0.6)

        if len(classIds) != 0:
            person_bboxes = [box for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox) if classId == 1]

            if person_bboxes:
                x, y, w, h = union_bounding_box(person_bboxes)

                if smoothed_bbox is None:
                    smoothed_bbox = np.array([x, y, w, h], dtype=np.float32)
                else:
                    smoothed_bbox = smoothing_factor * smoothed_bbox + (1 - smoothing_factor) * np.array([x, y, w, h], dtype=np.float32)

                smoothed_bbox_int = smoothed_bbox.astype(int)
                cropped_frame = crop_and_resize(frame, smoothed_bbox_int[0], smoothed_bbox_int[1], smoothed_bbox_int[2], smoothed_bbox_int[3], output_dimensions)
            else:
                cropped_frame = cv2.resize(frame, output_dimensions)

            if debug:
                draw_boxes(frame, person_bboxes)

            out.write(cropped_frame)

            if debug:
                cv2.imshow('Debug Video', frame)
                cv2.imshow('Output Video', cropped_frame)

            # Check for the Escape key press to exit
            if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the Escape key
                break

    cap.release()
    out.release()
    if debug:
        cv2.destroyAllWindows()

def union_bounding_box(bboxes):
    x = min([box[0] for box in bboxes])
    y = min([box[1] for box in bboxes])
    x_w = max([box[0] + box[2] for box in bboxes])
    y_h = max([box[1] + box[3] for box in bboxes])
    return x, y, x_w - x, y_h - y

def crop_and_resize(frame, x, y, w, h, dims):
    cropped = frame[y:y+h, x:x+w]
    return cv2.resize(cropped, dims)

def draw_boxes(frame, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

if __name__ == "__main__":
    # VIDEO_SOURCE = "input.mp4"
    detect_and_focus_people(VIDEO_SOURCE, (640, 480), debug=True)