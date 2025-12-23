# `record.py` – Sign Language Video Recording Script

This script is used to record sign language gesture videos using a webcam.
It is a data collection utility within a larger Sign Language to Speech Conversion project.

The script captures short, fixed-length videos for each sign and stores them in a structured format suitable for further processing (e.g., landmark extraction, feature generation, model training).

Here is the script : 

## Purpose of the Script

record.py is responsible for:

- Recording **multiple samples** of each sign
- Maintaining a **fixed number of frames** per recording
- Saving gesture videos in an **organized directory structure**
- Providing **visual feedback** during recording

This script **does not perform recognition or prediction**.It is only used for **dataset creation**.

## Libraries Used

```python
import cv2 as cv
import numpy as np
import os
import time
```

| Library | Role                      |
|---------|---------------------------|
| `cv2 `    | Webcam access and video recording |
| `numpy`   | Storing sign labels       |
| `os`      | Folder creation and path handling |
| `time`    | Frame delay control       |

## Configuration Section

```python
DATA_PATH = os.path.join('MP_Data')
ACTIONS = np.array(['buds'])
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
FRAME_DELAY = 0.05
```

### Explanation

`DATA_PATH` : Base directory prepared for storing frame-wise or landmark data.

`ACTIONS` : List of sign labels to be recorded.
Each value represents one sign gesture.

`NO_SEQUENCES` : Number of video samples recorded per sign.

`SEQUENCE_LENGTH` : Number of frames recorded per video.
Ensures uniform input size for temporal models.

`FRAME_DELAY` : Delay between frames (controls recording speed).

## Dataset Folder Creation

```python
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
```

This prepares directories in the format:

```php-template
MP_Data/<sign>/<sequence_number>/
```

These folders are intended for:

- Extracted frames
- MediaPipe landmarks
- Feature vectors

Even though this script records videos, the structure supports later processing steps.

## Webcam Initialization

```python
cap = cv.VideoCapture(0)
```

- Opens the default system webcam
- A resizable window named "Recording" is created
- Live preview is shown during recording

## Recording Loop Overview

```python
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
```

- **Outer loop**: iterates over each sign
- **Inner loop**: records multiple samples for that sign

## Video Writer Setup

```python
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(video_path, fourcc, 30, (640, 480))
```

Each video:

- Is saved as .mp4
- Uses mp4v codec
- Has resolution 640 × 480
- Is recorded at 30 FPS

Videos are stored as:

```php-template
Recorded_Videos/<sign>/<sign>_<sequence>.mp4
```

## Pre-Recording Prompt

```python
cv.putText(frame, 'Starting <sign>', ...)
cv.waitKey(2000)
```

- Displays a message indicating the sign and sample number
- Waits 2 seconds so the signer can prepare
- Heps reduce inconsistent gesture starts

## Frame Capture Logic

```python
while frame_count < SEQUENCE_LENGTH:
```

For each frame:

- Frame is read from the webcam
- Frame is written to the video file
- Recording status text is displayed:
  - Sign name
  - Video number
  - Frame count
- Frame is shown in the preview window
- Delay is applied to control frame rate

## Manual Exit

```python
if cv.waitKey(1) & 0xFF == ord('q'):
    break
```

- Press q to stop recording early
- Useful during testing or incorrect gesture execution

## Saving the Video

```python
out.release()
```

- Finalizes the video file
- Confirms successful save in the console

## Cleanup

```python
cap.release()
cv.destroyAllWindows()
```

- Releases the webcam
- Closes all OpenCV windows
- Prevents resource locking issues

## Summary

`record.py`:

- Records labeled sign language videos
- Ensures fixed-length gesture sequences
- Saves clean, consistent data for downstream processing
- Acts as the data collection entry point of the project

This script is designed to be simple, reusable, and extensible within a larger sign language recognition system.