import cv2
import numpy as np

# Specify paths for the pre-trained model and video source
MODEL_PATH = 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WEIGHTS_PATH = 'models/frozen_inference_graph.pb'
VIDEO_SOURCE = 0  # Use 0 for webcam, or replace with a filepath or URL

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
    out = cv2.VideoWriter('video/output.avi', fourcc, 30.0, output_dimensions)

    # Initialize smoothing
    smoothed_bbox = None
    smoothing_factor = 0.9

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        classIds, confs, bbox = net.detect(frame, confThreshold=0.5)  # Increase confThreshold to improve stability


        # Ensure the outputs are numpy arrays
        classIds = np.array(classIds) if classIds is not None else np.array([])
        confs = np.array(confs) if confs is not None else np.array([])
        bbox = np.array(bbox) if bbox is not None else np.array([])

        # Filter for person class (classId == 1)
        # # Prepare the bounding box coordinates
        person_bboxes = [box for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox) if classId == 1]



        if person_bboxes:
            x, y, w, h = union_bounding_box(person_bboxes)

            if smoothed_bbox is None:
                smoothed_bbox = np.array([x, y, w, h], dtype=np.float32)
            else:
                current_bbox = np.array([x, y, w, h], dtype=np.float32)
                smoothed_bbox = smoothing_factor * smoothed_bbox + (1 - smoothing_factor) * current_bbox

            smoothed_bbox_int = smoothed_bbox.astype(int)
            centered_frame = crop_and_center(frame, *smoothed_bbox_int, output_dimensions)
        else:
            centered_frame = resize_and_pad(frame, output_dimensions)

        if debug:
            draw_boxes(frame, person_bboxes)

        out.write(centered_frame)

        if debug:
            cv2.imshow('Debug Video', frame)
            cv2.imshow('Output Video', centered_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Escape'
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

def crop_and_center(frame, x, y, w, h, output_dims):
    frame_height, frame_width = frame.shape[:2]
    target_width, target_height = output_dims

    center_x = x + w // 2
    center_y = y + h // 2

    new_w = min(frame_width, target_width)
    new_h = min(frame_height, target_height)

    new_x = max(0, int(center_x - new_w // 2))
    new_y = max(0, int(center_y - new_h // 2))

    if new_x + new_w > frame_width:
        new_x = frame_width - new_w
    if new_y + new_h > frame_height:
        new_y = frame_height - new_h

    cropped = frame[new_y:new_y+new_h, new_x:new_x+new_w]
    return resize_and_pad(cropped, output_dims)

def resize_and_pad(image, output_dims):
    target_width, target_height = output_dims
    image_height, image_width = image.shape[:2]
    
    scale = min(target_width / image_width, target_height / image_height)
    new_size = (int(image_width * scale), int(image_height * scale))
    resized_image = cv2.resize(image, new_size)

    dx = (target_width - resized_image.shape[1]) // 2
    dy = (target_height - resized_image.shape[0]) // 2

    output_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    output_image[dy:dy+resized_image.shape[0], dx:dx+resized_image.shape[1]] = resized_image


    return output_image

def draw_boxes(frame, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

if __name__ == "__main__":
    VIDEO_SOURCE = "video/sg.mp4"
    VIDEO_SOURCE = "video/jared.mp4"
    # VIDEO_SOURCE = 0
    # VIDEO_SOURCE = "video/input.mp4"
    # VIDEO_SOURCE = "video/walgreen.mp4"
    # detect_and_focus_people(VIDEO_SOURCE, (320, 180), debug=True)
    VIDEO_SOURCE = "video/input.mp4"
    detect_and_focus_people(VIDEO_SOURCE, (960, 540), debug=True)