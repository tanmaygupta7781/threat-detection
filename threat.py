import cv2
import torch
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
import time

# Device configuration
device = 'cpu'
print(f"Using device: {device}")

# Initialize YOLO model
model1 = YOLO("C:/Users/priya/objdec/gunmodel/model.pt")

# Initialize second model (e.g., custom PyTorch model or another YOLO variant)
model2 = YOLO("C:/Users/priya/objdec/best.pt")  # Replace with your second model path

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Object classes for YOLO
classNames = ['Weapon']

alert_triggered = False

# Email alert function
def send_email_alert():
    sender_email = "ganeshakabhagwaan@gmail.com"  
    receiver_email = "tanmaypandita111@gmail.com"  
    password = "rozc zguq arav lhqp"  

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

# Event logging function
def log_event(frame, object_name, frame_time):
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(frame_time))
    file_name = f"suspicious_event_{object_name}_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)  

    with open("event_log.txt", "a") as log_file:
        log_file.write(f"Suspicious {object_name} detected at {time.ctime(frame_time)}\n")
    print(f"Logged event for {object_name}")

# Main function
def main():
    global alert_triggered

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break

        # Perform inference with YOLO (model1)
        results1 = model1(frame, verbose=False)

        # Perform inference with the second model (model2)
        results2 = model2(frame, verbose=False)

        # Parse results from model1
        for r in results1:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = f"{classNames[cls]}"
                confidence = box.conf[0]

                # Draw bounding box and label for model1 results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Trigger alert for specific conditions
                if label in ["knife", "pistol", "gun"] and confidence > 0.8 and not alert_triggered:
                    print(f"Suspicious object detected by model1: {label}")
                    send_email_alert()
                    log_event(frame, label, time.time())
                    alert_triggered = True

        # Parse results from model2
        for r in results2:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = f"{classNames[cls]}"
                confidence = box.conf[0]

                # Draw bounding box and label for model2 results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Trigger alert for specific conditions
                if label in ["knife", "pistol", "gun"] and confidence > 0.8 and not alert_triggered:
                    print(f"Suspicious object detected by model2: {label}")
                    send_email_alert()
                    log_event(frame, label, time.time())
                    alert_triggered = True

        # Reset alert
        if alert_triggered:
            alert_triggered = False

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
