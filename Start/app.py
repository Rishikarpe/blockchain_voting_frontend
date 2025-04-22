import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import time
import random

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Variables for authentication
counter_right = 0
counter_left = 0
counter_center = 0
finger_numbers = [random.randint(1, 5) for _ in range(3)]
current_finger_index = 0
finger_verification_success = [False, False, False]
finger_verification_timer = 0
finger_hold_time = 30

# Camera setup
camera = cv2.VideoCapture(0)

# Landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Setup face and hand detectors
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1)

# Frame generator for video stream
def gen_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face and hands
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        if face_results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, face_results, draw=True)

        # Process hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    landmarks_list.append((id, cx, cy))

                fingers = [1 if landmarks_list[4][2] < landmarks_list[3][2] else 0 for i in range(5)]
                total_fingers = sum(fingers)
                cv2.putText(frame, f'Fingers: {total_fingers}', (frame.shape[1] - 220, frame.shape[0] - 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame as a byte-stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)

# Route to stream video to the frontend
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('kyc.html')

if __name__ == '__main__':
    app.run(debug=True)