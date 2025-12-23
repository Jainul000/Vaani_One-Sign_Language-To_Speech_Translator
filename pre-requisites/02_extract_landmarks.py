import cv2
import numpy as np
import os
import mediapipe as mp

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, pose, hands):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    pose_results = pose.process(image)
    hand_results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, (pose_results, hand_results)


def draw_landmarks(image, results):
    pose_results, hand_results = results
    # Draw pose connections
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    # Draw hand connections
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))


def extract_keypoints(results):
    pose_results, hand_results = results
    pose = np.array([[lm.x, lm.y, lm.z] for lm in
                     pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(
        33 * 3)

    hands = []
    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            hands.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        hands = np.array(hands).flatten()
        if len(hands) < 126:  # Pad if less than 2 hands
            hands = np.concatenate([hands, np.zeros(126 - len(hands))])
    else:
        hands = np.zeros(126)

    return np.concatenate([pose, hands])  # 33*3 + 126 = 225 features


# Configuration - must match record.py
DATA_PATH = os.path.join('MP_Data')
RECORDED_VIDEOS_PATH = os.path.join('Recorded_Videos')
actions = np.array(['buds', 'spray', 'grow'])  # Must match record.py's ACTIONS
no_sequences = 30  # Must match record.py's NO_SEQUENCES
sequence_length = 30  # Must match record.py's SEQUENCE_LENGTH

# Create dataset directories
for action in actions:
    for seq in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)))
        except:
            pass

# Process recorded videos
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    for action in actions:
        for sequence in range(no_sequences):
            video_path = os.path.join(RECORDED_VIDEOS_PATH, action, f"{action}_{sequence}.mp4")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                continue

            frame_count = 0
            while frame_count < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video {video_path} at frame {frame_count}")
                    break

                # Process frame
                image, results = mediapipe_detection(frame, pose, hands)
                draw_landmarks(image, results)

                # Extract and save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_count))
                np.save(npy_path, keypoints)

                # Display processing
                cv2.putText(image, f"Processing: {action} {sequence + 1}/{no_sequences}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Frame: {frame_count + 1}/{sequence_length}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Processing Videos', image)

                frame_count += 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            print(f"Processed: {video_path}")

cv2.destroyAllWindows()
print("All videos processed and landmarks extracted!")