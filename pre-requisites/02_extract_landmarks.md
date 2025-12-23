# `extract_keypoints.py` – MediaPipe Landmark Extraction Script

This script processes previously recorded sign language videos and extracts pose and hand landmarks using **MediaPipe**.
The extracted landmarks are saved as **NumPy** (`.npy`) files and serve as model-ready input features for training sign language recognition models.

This script is used after record.py in the larger Sign Language to Speech Conversion project.

Here is the script : 

## Purpose of the Script

This script is responsible for:

- Reading recorded sign language videos
- Detecting **pose and hand landmarks** using MediaPipe
- Converting landmarks into **fixed-length numerical feature vectors**
- Saving frame-wise keypoints in a structured dataset format
- Visualizing detected landmarks during processing

It **does not perform training or prediction**.
Its sole purpose is **feature extraction**.

## Libraries Used

```python
import cv2
import numpy as np
import os
import mediapipe as mp
```

| Library | Role                      |
|---------|---------------------------|
| `cv2`     | Video reading and visualization |
| `numpy`   | Feature vector creation and storage       |
| `os`      | File and directory management |
| `mediapipe`   | Pose and hand landmark detection      |

## MediaPipe Setup

```python
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
```

### Explanation

- **Pose model**: Extracts full-body landmarks (33 points)
- **Hands model**: Extracts hand landmarks (21 points per hand)
- **Drawing utils**: Used only for visualization

Both pose and hands are processed together to capture **complete sign gestures**.

## Helper Functions

### 1. `mediapipe_detection()`:
```python
def mediapipe_detection(image, pose, hands):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    pose_results = pose.process(image)
    hand_results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, (pose_results, hand_results)
```


- Converts image to RGB
- Runs MediaPipe pose and hand detection
- Converts image back to BGR
- Returns processed image and detection results

This function standardizes MediaPipe inference for each frame.

### 2. `draw_landmarks()`:
```python
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
```

- Draws detected pose landmarks and connections
- Draws hand landmarks for up to two hands
- Used only for visualization during processing

This helps visually verify that landmarks are being detected correctly.

### 3. `extract_keypoints()`:
```python
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
```

This function converts MediaPipe results into a single numeric feature vector:

- Pose: 33 landmarks × 3 coordinates = 99 values
- Hands: 2 hands × 21 landmarks × 3 coordinates = 126 values
- Total features per frame: 225

If pose or hands are not detected:

- Missing landmarks are replaced with zeros
- Output shape remains consistent

```python
return np.concatenate([pose, hands])
```

This consistency is critical for training deep learning models.

## Configuration

```python
DATA_PATH = 'MP_Data'
RECORDED_VIDEOS_PATH = 'Recorded_Videos'
actions = ['buds', 'spray', 'grow']
no_sequences = 30
sequence_length = 30
```

### ⚠️ Important
These values must match `record.py` exactly to ensure correct **alignment between videos and extracted keypoints**.

## Dataset Structure

Extracted keypoints are saved as:

```php-template
MP_Data/<action>/<sequence>/<frame>.npy
```

Example:

```swift
MP_Data/buds/0/0.npy
MP_Data/buds/0/1.npy
...
```

Each `.npy` file contains a **225-dimensional feature vector** for one frame.

## Video Processing Loop

The script loops through:

- Each action
- Each sequence
- Each frame within the video

For every frame:

1. Read frame from video
2. Run MediaPipe detection
3. Draw landmarks (for visualization)
4. Extract keypoints
5. Save keypoints as `.npy`
6. Display processing status

Processing stops once the configured frame count is reached.

## Visualization During Processing

During execution, a window displays:

- Detected pose and hand landmarks
- Current action and sequence number
- Current frame number

This allows real-time verification of detection quality.

Press `q` at any time to stop processing early.

## Frame Capture Logic

```python
cap.release()
cv2.destroyAllWindows()
```

- Releases video resources
- Closes all OpenCV windows
- Prints confirmation message

## Summary

`extract_landmarks.py`:

- Converts raw sign language videos into numerical landmark data
- Produces consistent, model-ready feature vectors
- Bridges the gap between video recording and model training
- Is a critical preprocessing step in the Sign Language to Speech project

It is designed to be deterministic, structured, and easy to extend.