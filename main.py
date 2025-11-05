import cv2
import mediapipe as mp
import math
import pyttsx3
import winsound  # For buzzer sound on Windows

# Initialize mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to calculate posture accuracy
def get_posture_accuracy(angle):
    if angle >= 180:
        return 100
    elif angle >= 160:
        return round(((angle - 160) / 20) * 100, 1)
    else:
        return round((angle / 160) * 60, 1)

cap = cv2.VideoCapture(0)
feedback_spoken = None

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            left_angle = calculate_angle(left_shoulder, left_hip, right_hip)
            right_angle = calculate_angle(right_shoulder, right_hip, left_hip)
            avg_back_angle = (left_angle + right_angle) / 2
            accuracy = get_posture_accuracy(avg_back_angle)

            # Determine feedback
            if 150 <= avg_back_angle <= 185:
                feedback = "Good Posture 😊"
                color = (0, 255, 0)
                if feedback_spoken != "good":
                    engine.say("Good posture, keep it up")
                    engine.runAndWait()
                    feedback_spoken = "good"

            elif 140 <= avg_back_angle < 150 or 185 < avg_back_angle <= 195:
                feedback = "Neutral Posture ⚠️"
                color = (0, 255, 255)
                if feedback_spoken != "neutral":
                    engine.say("Neutral posture. Adjust a little")
                    engine.runAndWait()
                    feedback_spoken = "neutral"

            else:
                feedback = "Bad Posture! Straighten your back! ❌"
                color = (0, 0, 255)
                if feedback_spoken != "bad":
                    engine.say("Bad posture detected, straighten your back")
                    engine.runAndWait()
                    winsound.Beep(1000, 600)
                    feedback_spoken = "bad"

            # Display feedback text
            cv2.putText(image, f"{feedback}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            cv2.putText(image, f"Posture Accuracy: {accuracy}%", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            # ======= NEW ADDITION: Accuracy Progress Bar =======
            bar_x, bar_y = 30, 130
            bar_width, bar_height = 300, 20
            fill_width = int((accuracy / 100) * bar_width)

            # Bar background
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Bar fill (green/yellow/red)
            bar_color = (0, 255, 0) if accuracy >= 80 else (0, 255, 255) if accuracy >= 50 else (0, 0, 255)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
            # Bar border
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            # ==================================================

        except Exception:
            pass

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('PosePerfect - AI Voice & Buzzer Posture Corrector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

