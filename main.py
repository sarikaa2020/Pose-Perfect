import cv2
import mediapipe as mp
import math
import pyttsx3
import winsound
import numpy as np

# Initialize mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Setup
cap = cv2.VideoCapture(0)
feedback_spoken = None
reference_angle = None  # To store calibrated "good posture" angle

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get important landmarks (left ear, shoulder, hip)
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # Calculate neck-back angle
            angle = calculate_angle(left_ear, left_shoulder, left_hip)

            if reference_angle is not None:
                deviation = abs(angle - reference_angle)
                posture_accuracy = max(0, min(100, 100 - deviation * 2))  # Clamp between 0–100

                # Posture feedback based on deviation
                if deviation < 5:
                    feedback = "Perfect Posture 😊"
                    color = (0, 255, 0)
                    if feedback_spoken != "good":
                        engine.say("Perfect posture, keep it up")
                        engine.runAndWait()
                        feedback_spoken = "good"
                elif deviation < 10:
                    feedback = "Slightly Off ⚠️"
                    color = (0, 255, 255)
                    if feedback_spoken != "neutral":
                        engine.say("Adjust posture slightly")
                        engine.runAndWait()
                        feedback_spoken = "neutral"
                else:
                    feedback = "Bad Posture! ❌"
                    color = (0, 0, 255)
                    if feedback_spoken != "bad":
                        engine.say("Bad posture detected, straighten your back")
                        engine.runAndWait()
                        winsound.Beep(1000, 500)
                        feedback_spoken = "bad"

                # Draw feedback and posture accuracy
                cv2.putText(image, f"{feedback}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
                cv2.putText(image, f"Posture Accuracy: {posture_accuracy:.1f}%", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Sit straight & Press 'C' to Calibrate Good Posture", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception:
            pass

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show camera feed
        cv2.imshow('PosePerfect - AI Voice & Buzzer Posture Corrector', image)

        # Handle key press (single waitKey for both)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            reference_angle = angle
            engine.say("Good posture calibrated successfully")
            engine.runAndWait()
            print(f"✅ Calibrated good posture angle: {reference_angle:.2f}°")
        elif key == ord('q'):
            print("🛑 Exiting PosePerfect...")
            break

cap.release()
cv2.destroyAllWindows()
engine.stop()
