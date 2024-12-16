import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("./train/weights/best.pt")

# Replace 'http://<IP>:<port>/video' with the actual URL provided by your phone's camera app
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)  # Pass the frame to the model for detection

    # Filter results based on confidence threshold
    confidence_threshold = 0.85  # Update the confidence threshold to 0.85
    filtered_boxes = [box for box in results[0].boxes if box.conf[0] > confidence_threshold]  # Access confidence

    # Check if there are any results to plot
    if filtered_boxes:
        frame = results[0].plot(show=False, conf=True)  # Draw the results on the frame
        
    else:
        frame = frame  # Keep the original frame

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()