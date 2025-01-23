import cv2
import numpy as np

# Specify paths for the pre-trained YOLO model and configuration
CONFIG_PATH = 'yolov3.cfg'
WEIGHTS_PATH = 'yolov3.weights'
CLASSES_PATH = 'coco.names'
VIDEO_SOURCE = 0  # 0 for webcam, or replace with a filepath or URL

def load_yolo():
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# def get_output_layers(net):
#     layer_names = net.getLayerNames()
#     return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def get_output_layers(net):
    # Fetch layer names, then get the output layers as required by YOLO
    layer_names = net.getLayerNames()
    
    # The `getUnconnectedOutLayers()` function often returns 1-based indices, so we subtract 1.
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def draw_bounding_box(frame, class_id, confidence, x, y, x_plus_w, y_plus_h, debug):
    if debug:
        label = str(class_id)
        color = (0, 255, 0)  # Green for person
        cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_and_focus_people(video_source, output_dimensions=(640, 480), conf_threshold=0.5, nms_threshold=0.4, debug=False):
    net = load_yolo()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    classes = None
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    person_class_id = classes.index('person')

    # Assuming initialization of the VideoWriter is done here:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, output_dimensions)

    if not out.isOpened():
        print("Error: Could not open the video writer.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([int(x), int(y), int(w), int(h)])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        person_boxes = [boxes[i] for i in indices if class_ids[i] == person_class_id]

        if person_boxes:
            x, y, w, h = union_bounding_box(person_boxes)
            cropped_frame = crop_and_resize(frame, x, y, w, h, output_dimensions)
        else:
            cropped_frame = cv2.resize(frame, output_dimensions)

        out.write(cropped_frame)

        if debug:
            draw_boxes(frame, boxes, class_ids, confidences, indices, person_class_id)

            cv2.imshow('Debug Video', frame)
            cv2.imshow('Output Video', cropped_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    cropped = frame[max(0, y):y+h, max(0, x):x+w]
    return cv2.resize(cropped, dims)

def draw_boxes(frame, boxes, class_ids, confidences, indices, person_class_id):
    for i in indices:
        box = boxes[i]
        if class_ids[i] == person_class_id:
            draw_bounding_box(frame, class_ids[i], confidences[i], box[0], box[1], box[0] + box[2], box[1] + box[3], debug=True)

if __name__ == "__main__":
    detect_and_focus_people(VIDEO_SOURCE, (640, 480), debug=True)