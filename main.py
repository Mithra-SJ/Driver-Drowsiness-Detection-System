import cv2
import mediapipe as mp
import time
import winsound
from collections import deque
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Start video capture
cap = cv2.VideoCapture(0)

# Rolling window for EAR approximation
EAR_QUEUE = deque(maxlen=5)  # Last 5 frames
EAR_THRESHOLD = 0.5          # Eye closure threshold (adjust if needed)
ALERT_DURATION = 10           # seconds for alert display

last_alert_time = 0
alert_active = False

# Make window resizable
cv2.namedWindow("Driver Monitor", cv2.WINDOW_NORMAL)

def play_buzzer():
    # Play buzzer in a non-blocking way
    winsound.Beep(1000, 800)

while True:
    success, frame = cap.read()
    if not success:
        break

    display_frame = frame.copy()          # Keep full-size for display
    proc_frame = cv2.resize(frame, (320, 240))  # Smaller frame for faster processing
    rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    current_time = time.time()

    drowsy = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye landmarks
            left_eye = [face_landmarks.landmark[i] for i in range(474, 478)]
            right_eye = [face_landmarks.landmark[i] for i in range(469, 473)]

            avg_left_y = sum([p.y for p in left_eye]) / len(left_eye)
            avg_right_y = sum([p.y for p in right_eye]) / len(right_eye)

            EAR_QUEUE.append((avg_left_y + avg_right_y) / 2)

            # Average over rolling window
            if len(EAR_QUEUE) == EAR_QUEUE.maxlen:
                avg_ear = sum(EAR_QUEUE) / len(EAR_QUEUE)
                if avg_ear > EAR_THRESHOLD:
                    drowsy = True

    # Trigger alert
    if drowsy:
        alert_active = True
        last_alert_time = current_time
        print("Distraction Detected!")
        play_buzzer()

    # Display alert for ALERT_DURATION seconds
    if alert_active and (current_time - last_alert_time < ALERT_DURATION):
        cv2.putText(display_frame, "Distraction Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        alert_active = False

    # Show camera feed
    cv2.imshow("Driver Monitor", display_frame)

    # Press 'q' to safely exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
