# 💤 Drowsiness Detection System

A real-time drowsiness detection system for drivers using computer vision and machine learning. This project helps prevent accidents by monitoring the driver's eye activity and alerting them or emergency contacts when signs of sleepiness are detected.

## 🚀 Features

- 🔍 Real-time eye aspect ratio (EAR) detection using dlib landmarks
- 🧠 Eye state classification using HOG + SVM + EAR thresholds
- 📢 Audio alerts with short and long beeps for drowsy or sleepy states
- 📧 Automatic email alerts with image snapshots to emergency contact
- 🎥 Live webcam feed via Flask web server
- 🧑‍💻 Frontend with user input (username, vehicle number, emergency email)
- 📊 Dashboard to display real-time status updates and system feedback

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Python, Flask
- **Libraries:** OpenCV, dlib, imutils, playsound, smtplib
- **ML Model:** HOG + SVM (trained offline), EAR threshold-based detection


## 🧠 How It Works

1. **Eye Detection:** Uses dlib’s facial landmarks to detect eyes.
2. **EAR Calculation:** Computes Eye Aspect Ratio (EAR) to determine eye closure.
3. **State Classification:**
   - EAR > 0.25: 🟢 Active
   - EAR 0.21–0.25: 🟡 Drowsy
   - EAR < 0.21: 🔴 Sleepy
4. **Alerts:**
   - Drowsy: short beep
   - Sleepy: long beep + email alert with snapshot
5. **Flask Dashboard:** Displays live video, user inputs, and detection status.

## 🧪 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sangeethagireesh/drowsiness-detection.git
   cd drowsiness-detection




