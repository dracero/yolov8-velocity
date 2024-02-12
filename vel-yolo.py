import cv2
import torch
from view import ViewTransformer
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.35
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280
############ HAY QUE VER CÓMO DEFINIR ESTO
SOURCE = np.array([
    [13, 8],
    [23, 9],
    [50, 21],
    [-5, 21]
])

TARGET_WIDTH = 5
TARGET_HEIGHT = 10

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])
############ HAY QUE VER CÓMO DEFINIR ESTO (ACÁ termina)
#El mismo codigo pero con webcam en tiempo real
model = YOLO(MODEL_NAME)


# Use OpenCV to capture video from the webcam
cap = cv2.VideoCapture(0)

# Get video info from the webcam
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=frame_rate, track_thresh=CONFIDENCE_THRESHOLD
)

# annotators configuration
thickness = sv.calculate_dynamic_line_thickness(
    resolution_wh=(frame_width, frame_height)
)
text_scale = sv.calculate_dynamic_text_scale(
    resolution_wh=(frame_width, frame_height)
)
bounding_box_annotator = sv.BoundingBoxAnnotator(
    thickness=thickness
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=int(frame_rate * 2),
    position=sv.Position.BOTTOM_CENTER
)

polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=(frame_width, frame_height)
)

coordinates = defaultdict(lambda: deque(maxlen=int(frame_rate)))

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# loop over frames from the webcam
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # filter out detections by class and confidence
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    #detections = detections[detections.class_id != 0]
   
    # filter out detections outside the zone
    #detections = detections[polygon_zone.trigger(detections)]

    # refine detections using non-max suppression
    detections = detections.with_nms(IOU_THRESHOLD)

    # pass detection through the tracker
    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchors_coordinates(
        anchor=sv.Position.BOTTOM_CENTER
    )

    # calculate the detections position inside the target RoI
    points = view_transformer.transform_points(points=points).astype(int)
   
    # store detections position
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    # format labels
    labels = []

    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < frame_rate / 2:
            labels.append(f"#{tracker_id}")
        else:
            # calculate speed
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / frame_rate
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    # annotate frame
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    # Display the resulting frame
    cv2.imshow('frame', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()


