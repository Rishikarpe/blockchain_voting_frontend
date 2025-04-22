import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
import random

# Variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
counter_right = 0
counter_left = 0
counter_center = 0
total_fingers = 0
gaze_sequence = []
auth_stage = 0  # Authentication stage
auth_success = False  # Authentication status
last_direction = ""  # Last detected eye direction
direction_confirmed = False  # Flag to confirm direction change
look_left_confirmed = False  # Flag to confirm looking left
look_right_confirmed = False  # Flag to confirm looking right
exercise_completed = False  # Flag to confirm gaze exercise completion
finger_auth_completed = False  # Flag to confirm finger authentication
auth_message = "Look LEFT then RIGHT"  # Initial instruction

# New variables for enhanced finger verification
finger_numbers = [random.randint(1, 5) for _ in range(3)]  # Generate 3 random finger counts
current_finger_index = 0  # Track which finger verification we're on
finger_verification_success = [False, False, False]  # Track success for each verification
finger_verification_timer = 0  # Timer to confirm finger count
finger_hold_time = 30  # Frames to hold the finger count (about 1 second at 30fps)

# Constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX
DIRECTION_FRAMES_THRESHOLD = 1  # Frames needed to confirm a direction

# Face bounder indices 
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Finger tip landmarks
finger_tips_ids = [4, 8, 12, 16, 20]

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Camera setup
camera = cv.VideoCapture(0)
_, frame = camera.read(1)
frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
img_height, img_width = frame.shape[:2]
print(f"Frame dimensions: {img_height}, {img_width}")

# Video recording setup 
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('authentication_session.mp4', fourcc, 30.0, (img_width, img_height))

# Landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) 
                  for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Euclidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)*2 + (y1 - y)*2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eye horizontal points
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # Right eye vertical points
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye horizontal points
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # Left eye vertical points
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    # Calculate distances
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    # Calculate ratio
    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio 

# Eyes Extractor function
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)

    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155
    
    # Right eye bounds
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # Left eye bounds
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    return cropped_right, cropped_left

# Eye Position Estimator 
def positionEstimator(cropped_eye):
    if cropped_eye is None or cropped_eye.size == 0:
        return "UNKNOWN", [128, 128, 128]
        
    h, w = cropped_eye.shape
    if h == 0 or w == 0:
        return "UNKNOWN", [128, 128, 128]
        
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    piece = int(w / 3)
    if piece <= 0:
        return "UNKNOWN", [128, 128, 128]

    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]
    
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# Pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    global counter_right, counter_left, counter_center
    
    # Count black pixels in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    
    eye_parts = [right_part, center_part, left_part]
    
    # Get the position with maximum black pixels
    max_index = eye_parts.index(max(eye_parts))
    
    # Set position and color
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [0, 255, 0]
        counter_right += 1
        counter_left = 0
        counter_center = 0
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [255, 255, 0]
        counter_center += 1
        counter_right = 0
        counter_left = 0
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [0, 0, 255]
        counter_left += 1
        counter_right = 0
        counter_center = 0
    else:
        pos_eye = "UNKNOWN"
        color = [128, 128, 128]
        
    return pos_eye, color

# Determine consistent gaze direction
def get_consistent_direction(right_pos, left_pos):
    # If both eyes are looking in the same direction, return that direction
    # Otherwise, prioritize the direction with more confidence
    if right_pos == left_pos:
        return right_pos
    elif right_pos == "RIGHT" or left_pos == "RIGHT":
        if counter_right > DIRECTION_FRAMES_THRESHOLD:
            return "RIGHT"
    elif right_pos == "LEFT" or left_pos == "LEFT":
        if counter_left > DIRECTION_FRAMES_THRESHOLD:
            return "LEFT"
    elif right_pos == "CENTER" or left_pos == "CENTER":
        if counter_center > DIRECTION_FRAMES_THRESHOLD:
            return "CENTER"
    return "UNKNOWN"

# Draw authentication status section
def draw_auth_status(frame, frame_width, frame_height):
    global auth_message, exercise_completed, finger_auth_completed, auth_success, finger_verification_success, current_finger_index, finger_numbers
    
    # Draw authentication status box at the top
    cv.rectangle(frame, (20, 10), (frame_width - 20, 150), (40, 40, 40), -1)
    
    # Draw progress bars
    # Gaze exercise progress
    cv.rectangle(frame, (30, 50), (frame_width//2 - 30, 70), (50, 50, 50), -1)
    if look_left_confirmed:
        cv.rectangle(frame, (30, 50), (frame_width//4, 70), (0, 200, 0), -1)
    if look_right_confirmed:
        cv.rectangle(frame, (frame_width//4, 50), (frame_width//2 - 30, 70), (0, 200, 0), -1)
    
    # Finger authentication progress - now divided into 3 sections
    finger_section_width = (frame_width - 30 - (frame_width//2 + 30)) // 3
    for i in range(3):
        # Draw the background for each section
        section_start = frame_width//2 + 30 + (finger_section_width * i)
        section_end = section_start + finger_section_width
        cv.rectangle(frame, (section_start, 50), (section_end, 70), (50, 50, 50), -1)
        
        # Fill in green if this verification is complete
        if finger_verification_success[i]:
            cv.rectangle(frame, (section_start, 50), (section_end, 70), (0, 200, 0), -1)
    
    # Display instructions and status
    cv.putText(frame, "Enhanced Authentication System", (frame_width//2 - 180, 35), FONTS, 0.9, (255, 255, 255), 2)
    
    # Left side - Eye movement exercise
    cv.putText(frame, "Eye Exercise", (30, 40), FONTS, 0.6, (200, 200, 200), 1)
    
    # Right side - Finger authentication
    cv.putText(frame, "Finger Authentication (3 Steps)", (frame_width//2 + 30, 40), FONTS, 0.6, (200, 200, 200), 1)
    
    # Display current instruction
    cv.putText(frame, auth_message, (frame_width//2 - 150, 100), FONTS, 0.7, (255, 255, 255), 2)
    
    # For finger authentication, display the number to show
    if exercise_completed and not finger_auth_completed:
        cv.putText(frame, f"Step {current_finger_index + 1}/3: Show {finger_numbers[current_finger_index]} finger(s)", 
                  (frame_width//2 - 150, 130), FONTS, 0.7, (0, 255, 255), 2)
    
    # Display success message
    if auth_success:
        overlay = frame.copy()
        cv.rectangle(overlay, (frame_width//2 - 200, frame_height//2 - 50), 
                    (frame_width//2 + 200, frame_height//2 + 50), (0, 100, 0), -1)
        cv.putText(overlay, "Authentication Successful", (frame_width//2 - 180, frame_height//2 + 10), 
                  FONTS, 0.9, (255, 255, 255), 2)
        # Apply overlay with transparency
        cv.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

# Initialize MediaPipe solutions
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    with mp_hands.Hands(max_num_hands=1) as hands:
        start_time = time.time()
        
        while True:
            frame_counter += 1
            ret, frame = camera.read()
            if not ret: 
                break
            
            # Resize and flip the frame for mirror view
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame = cv.flip(frame, 1)  # Mirror view
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Process face mesh
            face_results = face_mesh.process(rgb_frame)
            # Process hands
            hand_results = hands.process(rgb_frame)
            
            # ---------- Authentication Logic ----------
            # Stage 1: Eye movement exercise
            # Stage 2: Finger count verification (now 3 steps)
            # Stage 3: Authentication success
            
            # ---------- Face Mesh Detection ----------
            current_gaze = "UNKNOWN"
            if face_results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, face_results, False)
                
                # Draw eye contours
                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)

                # Extract eye regions for gaze detection
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)

                # Determine eye positions
                eye_position_right, r_color = positionEstimator(crop_right)
                eye_position_left, l_color = positionEstimator(crop_left)
                
                # Display eye positions
                cv.putText(frame, f'R: {eye_position_right}', (40, frame_height - 60), FONTS, 0.6, r_color, 1)
                cv.putText(frame, f'L: {eye_position_left}', (40, frame_height - 30), FONTS, 0.6, l_color, 1)
                
                # Get consistent gaze direction across both eyes
                current_gaze = get_consistent_direction(eye_position_right, eye_position_left)
                
                # Eye exercise authentication logic
                if not exercise_completed:
                    if current_gaze == "LEFT" and counter_left > DIRECTION_FRAMES_THRESHOLD:
                        look_left_confirmed = True
                        if not look_right_confirmed:
                            auth_message = "Now look RIGHT"
                    
                    if look_left_confirmed and current_gaze == "RIGHT" and counter_right > DIRECTION_FRAMES_THRESHOLD:
                        look_right_confirmed = True
                        exercise_completed = True
                        auth_message = "Eye exercise completed! Now for finger verification."
                
            # ---------- Hand Detection ----------
            if hand_results.multi_hand_landmarks and exercise_completed and not finger_auth_completed:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    landmarks_list = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                        landmarks_list.append((id, cx, cy))

                    fingers = []

                    # Thumb (different condition - based on horizontal position)
                    if landmarks_list[finger_tips_ids[0]][1] < landmarks_list[finger_tips_ids[0] - 1][1]:  # Adjusted for mirrored view
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other fingers (based on vertical position)
                    for tip_id in finger_tips_ids[1:]:
                        if landmarks_list[tip_id][2] < landmarks_list[tip_id - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    total_fingers = sum(fingers)

                    # Display finger count
                    cv.putText(frame, f'Fingers: {total_fingers}', (frame_width - 220, frame_height - 30),
                                FONTS, 0.7, (0, 255, 0), 2)

                    # Enhanced finger verification - verify each number in sequence
                    if total_fingers == finger_numbers[current_finger_index]:
                        finger_verification_timer += 1
                        # Display a progress bar for holding the position
                        progress_percent = min(finger_verification_timer / finger_hold_time, 1.0)
                        progress_width = int(150 * progress_percent)
                        cv.rectangle(frame, (frame_width - 220, frame_height - 20), 
                                    (frame_width - 220 + progress_width, frame_height - 10), (0, 255, 0), -1)
                        
                        # If held long enough, confirm this step
                        if finger_verification_timer >= finger_hold_time:
                            finger_verification_success[current_finger_index] = True
                            current_finger_index += 1
                            finger_verification_timer = 0  # Reset timer for next verification
                            
                            # Check if all verifications are complete
                            if current_finger_index >= len(finger_numbers):
                                finger_auth_completed = True
                                auth_message = "Authentication successful!"
                                auth_success = True
                            else:
                                auth_message = f"Great! Now show {finger_numbers[current_finger_index]} finger(s)"
                    else:
                        # Reset timer if showing wrong number of fingers
                        finger_verification_timer = 0

                    # Draw hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw authentication status information
            draw_auth_status(frame, frame_width, frame_height)
            
            # FPS calculation and display
            end_time = time.time() - start_time
            fps = frame_counter / end_time
            cv.putText(frame, f'FPS: {round(fps, 1)}', (frame_width - 120, frame_height - 60), FONTS, 0.7, (0, 255, 255), 2)

            # Save and display the frame
            out.write(frame)
            cv.imshow('Video Authentication', frame)

            key = cv.waitKey(2)
            if key == ord('q') or key == ord('Q'):
                break
            # Reset authentication with 'r' key
            elif key == ord('r') or key == ord('R'):
                # Reset all authentication variables
                look_left_confirmed = False
                look_right_confirmed = False
                exercise_completed = False
                finger_auth_completed = False
                auth_success = False
                counter_right = 0
                counter_left = 0
                counter_center = 0
                # Reset finger verification variables
                finger_numbers = [random.randint(1, 5) for _ in range(3)]
                current_finger_index = 0
                finger_verification_success = [False, False, False]
                finger_verification_timer = 0
                auth_message = "Look LEFT then RIGHT"

        cv.destroyAllWindows()
        camera.release()
        out.release()