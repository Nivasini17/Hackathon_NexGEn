import numpy as np  # Imports the NumPy library for numerical operations
import cv2  # Imports the OpenCV library for computer vision tasks

# Parameters
whT = 608  # Width and height of the input image for the YOLO model
confThreshold = 0.5  # Confidence threshold for bounding box detection
nmsThreshold = 0.4  # Non-Maximum Suppression (NMS) threshold

# Load class names from coco.names file
with open('C:/Users/HP-INDIA/OneDrive/Desktop/Nivasini/HFH/coco.names') as f:
    classes = f.read().strip().split('\n')

# Load YOLO model
modelConfig = 'C:/Users/HP-INDIA/OneDrive/Desktop/Nivasini/HFH/yolov3.cfg'  # Path to the YOLOv3 configuration file
modelWeights = 'C:/Users/HP-INDIA/OneDrive/Desktop/Nivasini/HFH/yolov3.weights'  # Path to the YOLOv3 weights file

net = cv2.dnn.readNetFromDarknet(YOLOv3.cfg, YOLOv3.weights)  # type: ignore # Load YOLO model
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Set preferred backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Set preferred target device

# Get the names of the output layers
def getOutputsNames(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = f'{classes[classId]}: {conf:.2f}' if classes else f'{conf:.2f}'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outp):
    frameHeight, frameWidth = frame.shape[:2]
    classIds = []
    confidences = []
    boxes = []

    for out in outp:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from webcam")
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(getOutputsNames(net))
    postprocess(frame, outputs)
    
    cv2.imshow('Webcam Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()