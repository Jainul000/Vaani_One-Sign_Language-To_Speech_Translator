# `model_core.py` – Sign Detection & Sentence Generation Module

This script implements the core sign language detection logic for the Sign Language to Speech Conversion project.

It handles **video capture**, **keypoint extraction**, **model inference**, **sentence formation**, and **translation**.

All higher-level interaction (UI, speech output, buttons) is handled elsewhere.

## Purpose of the Script

This script is responsible for:

- Capturing live video from the webcam
- Extracting pose and hand landmarks using MediaPipe
- Normalizing and preparing model input
- Running predictions using a trained deep learning model
- Building a meaningful sentence from detected signs
- Translating the sentence into Hindi
- Exposing helper functions for Flask APIs

It acts as the **intelligence layer** of the system.

## Libraries Used

```python
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from googletrans import Translator
```

| Library     | Role                                   |
| ----------- | ----------------------------------------- |
| OpenCV      | Webcam access and display                 |
| NumPy       | Numerical processing                      |
| MediaPipe   | Pose and hand landmark detection          |
| Keras       | Loading trained sign classification model |
| GoogleTrans | English → Hindi translation               |
| OS          | File and directory management             |


## Model & Preprocessing Setup

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'action5.h5')
PARAMS_PATH = os.path.join(BASE_DIR, 'models', 'preprocess_params5.npz')

model = None
params = None

def load_resources():
    global model, params
    if model is None:
        model = load_model(MODEL_PATH)
    if params is None:
        params = np.load(PARAMS_PATH)
    return model, params

model, params = load_resources()

mean, std = params['mean'], params['std']
```

- Loads a trained sequence-based sign recognition model
- Loads saved normalization parameters used during training
- Ensures real-time input matches training distribution

## Supported Actions

```python
actions = np.array([
    'use', 'crop', 'seed', 'area', 'for', 'sowing',
    'disease', 'in', 'lower', 'half', 'leaves',
    'fertilize', 'with', 'quantity', 'medicine',
    'spray', 'mixed', 'of', 'water'
])
```

Each model prediction maps to one word from this list.
These words are **incrementally combined to form sentences**.

## Global State Variables

```python
latest_sentence = ""
latest_hindi = ""
sentence = []
```

These variables:

- Store the most recent detected sentence
- Allow other modules (Flask app) to fetch or modify output
- Persist detection state across frames
  
## Keypoint Extraction

```python
def extract_keypoints(results):
    pose_results, hand_results = results
    pose = np.array([[lm.x, lm.y, lm.z] for lm in
                     pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33 * 3)

    hands = []
    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            hands.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        hands = np.array(hands).flatten()
        if len(hands) < 126:
            hands = np.concatenate([hands, np.zeros(126 - len(hands))])
    else:
        hands = np.zeros(126)

    return np.concatenate([pose, hands])
```
This function:

- Extracts 33 pose landmarks (x, y, z)
- Extracts up to 2 hands (21 landmarks each)
- Pads missing landmarks with zeros
- Returns a fixed-length vector (225 values)

This consistency is critical for LSTM-based models.

## MediaPipe Processing Pipeline

```python
def mediapipe_detection(image, pose, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    pose_results = pose.process(image_rgb)
    hand_results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, (pose_results, hand_results)
```

- Converts image to RGB
- Runs pose and hand detection
- Converts image back to BGR
- Returns:
  - Annotated frame
  - Detection results

## Sentence & Translation Management

```python
@app.route('/clear-last-word', methods=['POST'])
def clear_last_word():
    model_core.clear_last_word()
```

Removes the most recently detected word.

```python
def update_latest(sentence, hindi):
    global latest_sentence, latest_hindi
    latest_sentence = sentence
    latest_hindi = hindi
```

Stores the most recent English sentence and its Hindi translation.

## Fetch Output (Used by Flask)

```python
def get_latest_sentence():
    return latest_sentence

def get_latest_hindi():
    return latest_hindi
```

These functions allow external scripts to **poll real-time predictions**.

## Clearing Controls

#### Remove Last Word

```python
def clear_last_word():
    global latest_sentence, latest_hindi, sentence
    # Split the sentence into words and remove the last one
    words = latest_sentence.strip().split()
    if words:
        words.pop()  # Remove the last word
        sentence = words  # Update the global sentence list
        latest_sentence = " ".join(words)
        # Update the Hindi translation
        if latest_sentence:
            try:
                latest_hindi = translator.translate(latest_sentence, src='en', dest='hi').text
            except Exception as e:
                latest_hindi = "Translation Error"
        else:
            latest_hindi = ""
    else:
        latest_sentence = ""
        latest_hindi = ""
        sentence = []
```

- Removes the most recent word from the sentence
- Rebuilds the sentence
- Re-translates the updated sentence

Used for `undo functionality` in the UI.

#### Clear Entire Sentence

```python
def clear_all():
    global latest_sentence, latest_hindi, sentence
    latest_sentence = ""
    latest_hindi = ""
    sentence = []  # Reset the global sentence list
```

Resets all detection output and state.

## Real-Time Detection Loop

```python
def run_detection():
```

This is the main execution function, typically launched in a separate thread.

Core Workflow:
1. Capture frames from webcam
2. Extract pose and hand landmarks
3. Normalize keypoints
4. Maintain a rolling sequence of 30 frames
5. Run model prediction
6. Apply confidence thresholding
7. Append new actions to sentence
8. Translate and update output

## Confidence Filtering Logic

```python
threshold = 0.90
history_length = 15
```

- Predictions must remain stable over multiple frames
- Reduces false positives and jitter
- Ensures only confident signs are added to the sentence

## Sentence Construction

```python
if sentence[-1] != current_action:
    sentence.append(current_action)
```

- Prevents duplicate consecutive words
- Limits sentence length to avoid overflow
- Builds natural-looking output

## Live Video Display

```python
cv2.imshow('Sign Detection - Press Q to exit', image)
```

- Shows real-time camera feed
- Allows manual exit using `Q`

## Shutdown Handling

```python
cap.release()
cv2.destroyAllWindows()
```

Ensures clean release of camera and GUI resources.

## Summary

`model_core.py`:

- Converts **live human motion** into **structured model input**
- Performs **real-time sign classification**
- Builds **context-aware sentences**
- Translates output for accessibility
- Exposes clean interfaces for UI and speech layers

It is the **core engine** that makes sign-to-speech conversion possible.
