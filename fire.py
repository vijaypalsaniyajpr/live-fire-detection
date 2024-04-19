from ultralytics import YOLO
import cv2
import math
import winsound  # Importing winsound to play alert sound on Windows

# Running real time from webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the index of the desired camera
model = YOLO('fire.pt')

# Reading the classes
classnames = ['fire']

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with 
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if confidence > 40:
                    # Play alert sound 
                    winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 milliseconds

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
