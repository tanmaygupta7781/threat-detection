import cv2
import torch
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
import os
from flask import Flask, Response, render_template

# Initialize Flask app
app = Flask(__name__)

# Ensure GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLO model onto the GPU
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
              "teddy bear", "hair drier", "toothbrush"
              ]

# Alert function to send email
def send_email_alert():
    sender_email = os.getenv('SENDER_EMAIL')
    receiver_email = os.getenv('RECEIVER_EMAIL')
    password = os.getenv('EMAIL_PASSWORD')

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

# Flag to prevent multiple alerts
alert_triggered = False

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

            results = model(frame,verbose=False)  # Run YOLO detection

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
                        send_email_alert()
                        alert_triggered = True  # Set flag to avoid multiple alerts

            # Reset alert flag if no objects are detected
            if alert_triggered and not any(box.cls[0] == classNames.index('bottle') for r in results for box in r.boxes):
                alert_triggered = False

            # Encode the frame to send as a response
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame to the Flask route
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

