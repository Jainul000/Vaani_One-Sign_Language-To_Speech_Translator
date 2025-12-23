import cv2 as cv
import numpy as np
import os
import time

# === Configuration ===
DATA_PATH = os.path.join('MP_Data')   # For saving extracted frames (optional)
ACTIONS = np.array([
    # 'spray',
    # 'medicine',
    # 'mixture',
    # 'on',
    # 'fruit',
    # 'trees',
    # 'when',
    'buds'
    # 'starts',
    # 'to',
    # 'grow'
])
NO_SEQUENCES = 30         # Number of videos per action
SEQUENCE_LENGTH = 30      # Number of frames per video
FRAME_DELAY = 0.05        # Delay in seconds between frames (0.05s = ~20fps)

# === Create folders for data ===
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# === Start recording ===
cap = cv.VideoCapture(0)
cv.namedWindow('Recording', cv.WINDOW_NORMAL)
cv.resizeWindow('Recording', 900, 600)

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        # Setup video writer
        video_name = f"{action}_{sequence}.mp4"
        video_dir = os.path.join('Recorded_Videos', action)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, video_name)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(video_path, fourcc, 30, (640, 480))

        print(f"\nRecording for {action} - Video {sequence+1}/{NO_SEQUENCES}")

        # Display starting message
        frame = cap.read()[1]
        cv.putText(frame, f'Starting {action} {sequence+1}',
                   (120, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('Recording', frame)
        cv.waitKey(2000)  # 2-second delay before recording

        # === Frame capture loop ===
        frame_count = 0
        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            frame_count += 1

            # Display overlay info
            cv.putText(frame, f'Recording {action} Video {sequence+1}/{NO_SEQUENCES}',
                       (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame, f'Frame: {frame_count}/{SEQUENCE_LENGTH}',
                       (15, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)

            cv.imshow('Recording', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(FRAME_DELAY)  # Delay between frames

        out.release()
        print(f"Saved: {video_path}")

# === Cleanup ===
cap.release()
cv.destroyAllWindows()
