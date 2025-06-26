from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
import cv2
import numpy as np
import dlib
import winsound
import threading
import smtplib
import ssl
from email.message import EmailMessage
from imutils import face_utils
import os

app = Flask(__name__)
CORS(app)

# Global variables
user_data = {}
cap = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detection_running = False
current_state = "Active"
current_ear = 0.0

EYE_CLOSED_THRESHOLD = 0.18
EYE_PARTIALLY_CLOSED_THRESHOLD = 0.25

drowsy_count = 0
sleepy_count = 0
prev_state = "Active"

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def send_email(image_path):
    msg = EmailMessage()
    msg.set_content(f"🚨 Drowsiness Alert!\n\nDriver: {user_data['username']}\nVehicle: {user_data['vehicleNumber']}\n\nImmediate action required!")

    msg["Subject"] = "Drowsiness Alert!"
    msg["From"] = "sangeethasandhya468@gmail.com"
    msg["To"] = user_data["emergencyEmail"]

    with open(image_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename="drowsy_driver.jpg")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login("sangeethasandhya468@gmail.com", "uwhqzwojqfzavpqe")
            server.send_message(msg)
        print("✅ Emergency Email Sent!")
    except Exception as e:
        print("❌ Email Error:", e)

def detect_drowsiness():
    global cap, detection_running, drowsy_count, sleepy_count, prev_state, current_state, current_ear
    detection_running = True
    cap = cv2.VideoCapture(0)

    email_sent = False

    while detection_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        state = "Active"
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            current_ear = round(ear, 2)

            if ear < EYE_CLOSED_THRESHOLD:
                sleepy_count += 1
                drowsy_count = 0
                state = "Sleepy"
                if sleepy_count > 6:
                    if not email_sent:
                        img_path = "drowsy_driver.jpg"
                        cv2.imwrite(img_path, frame)
                        threading.Thread(target=send_email, args=(img_path,), daemon=True).start()
                        email_sent = True
                    if prev_state != "Sleepy":
                        winsound.Beep(1000, 2000)
                        prev_state = "Sleepy"

            elif EYE_CLOSED_THRESHOLD < ear < EYE_PARTIALLY_CLOSED_THRESHOLD:
                sleepy_count = 0
                drowsy_count += 1
                state = "Drowsy"
                if drowsy_count > 6 and prev_state != "Drowsy":
                    winsound.Beep(1500, 300)
                    prev_state = "Drowsy"

            else:
                sleepy_count = 0
                drowsy_count = 0
                state = "Active"
                if prev_state == "Sleepy":
                    email_sent = False
                prev_state = "Active"

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            color = (0, 255, 0) if state == "Active" else (0, 255, 255) if state == "Drowsy" else (0, 0, 255)
            cv2.putText(frame, f"State: {state}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        current_state = state

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detection_running = False

def generate_frames():
    global cap
    while detection_running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global user_data, detection_running
    if detection_running:
        return jsonify({"message": "Drowsiness detection is already running!"})

    user_data = request.json
    print("📩 Received User Data:", user_data)

    threading.Thread(target=detect_drowsiness, daemon=True).start()
    return jsonify({"message": "Drowsiness detection started!"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running
    detection_running = False
    return jsonify({"message": "Drowsiness detection stopped!"})

@app.route('/drowsiness_status')
def drowsiness_status():
    return jsonify({"state": current_state, "ear": current_ear})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
