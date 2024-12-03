import cv2
import torch
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
import time
import os
from flask import Flask, Response, render_template

# Initialize Flask app
app = Flask(__name__)

# Ensure GPU is available
device = 'cpu'  # Modify this to 'cuda' if GPU is available
print(f"Using device: {device}")

# Load the YOLO model onto the device (CPU or GPU)
model = YOLO("yolo-Weights/yolov8n.pt").to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Flag to prevent multiple alerts
alert_triggered = False

# Dictionary to track object detection timestamps (for anomaly detection)
detected_objects = {}

# Alert function to send email
def send_email_alert():
    sender_email = "ganeshakabhagwaan@gmail.com"  # Replace with your sender email
    receiver_email = "tanmaypandita111@gmail.com"  # Replace with the receiver email
    password = "rozc zguq arav lhqp"  # Replace with your app password (if using Gmail)

    message = MIMEText("Intruder detected!")
    message['Subject'] = "Security Alert"
    message['From'] = sender_email
    message['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email alert sent!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Log suspicious behavior (e.g., repeated object detections)
def log_event(frame, object_name, frame_time):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(frame_time))
    file_name = f"suspicious_event_{object_name}_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)  # Save the frame with the detected object

    with open("event_log.txt", "a") as log_file:
        log_file.write(f"Suspicious {object_name} detected at {time.ctime(frame_time)}\n")
    print(f"Logged event for {object_name}")

# Behavioral anomaly detection
def log_behavior(frame, cls_name, frame_time):
    """ Log suspicious behavior based on the frequency and occurrence of objects. """
    current_time = time.time()
    if cls_name not in detected_objects:
        detected_objects[cls_name] = []

    detected_objects[cls_name].append(frame_time)
    
    # Keep a time window of 5 minutes (300 seconds)
    detected_objects[cls_name] = [t for t in detected_objects[cls_name] if current_time - t < 300]
    
    # If an object appears more than 10 times in the last 5 minutes, log suspicious behavior
    if len(detected_objects[cls_name]) > 10:
        print(f"Suspicious behavior detected: {cls_name} appears frequently.")
        log_event(frame, cls_name, current_time)  # Log event with frame and class name

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for video stream
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global alert_triggered
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)  # Run YOLO detection

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])

                    # Draw bounding box and label on the frame
                    label = f"{classNames[cls]}"
                    color = (0, 255, 0)  # Green color for box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Trigger alert for specific detection (e.g., bottle)
                    if classNames[cls] == 'bottle' and not alert_triggered:
                        log_behavior(frame, classNames[cls], time.time())  # Log anomaly for bottle detection
                        
                        # Send email alert if bottle is detected
                        send_email_alert()
                        alert_triggered = True  # Prevent multiple alerts for the same detection

                    # If the detected object disappears, reset the alert trigger
                    if classNames[cls] != 'bottle' and alert_triggered:
                        alert_triggered = False

            # Convert frame to JPEG for streaming via Flask
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            
            # Yield frame in the format Flask expects for video streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Return the video stream as a response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
