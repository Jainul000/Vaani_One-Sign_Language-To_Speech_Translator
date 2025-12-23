# `app.py` – Flask Application & Speech Output Script

This script provides a Flask-based web interface for a Sign Language to Speech Conversion project.
It acts as the control and interaction layer between the user interface, the sign detection engine, and speech output.

The script does not perform sign detection itself.
Instead, it **orchestrates detection, retrieves predictions, and converts recognized sentences into speech**.

Here is the script : 

## Purpose of the Script

This script is responsible for:

- Hosting a **web interface** for **user interaction**
- Starting the sign detection process asynchronously
- Fetching the latest recognized words and sentences
- Clearing detected words or sentences
- Converting detected text into audible speech

It serves as the **application layer** of the project.

## Libraries Used

```python
from flask import Flask, render_template, request
from threading import Thread
import model_core
from gtts import gTTS
import tempfile
import pygame
import os
import time
```

| Library      | Role                                     |
| ------------ | ---------------------------------------- |
| `Flask`      | Web server and routing                   |
| `threading`  | Non-blocking background execution        |
| `model_core` | Core sign detection and prediction logic |
| `gTTS`       | Text-to-speech conversion                |
| `pygame`     | Audio playback                           |
| `tempfile`   | Temporary audio file handling            |

## Flask App Initialization

```python
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
```
Initializes the Flask application and sets up routing for browser and API interaction.

## Web Routes Overview

### Home Page

```python
@app.route('/')
def index():
    return render_template('2.html')
```

- Loads the main UI page
- Typically contains camera view, controls, and output display
  
### Start Sign Detection

```python
@app.route('/start-detection', methods=['POST'])
def start_detection():
    detection_thread = Thread(target=model_core.run_detection)
    detection_thread.daemon = True
    detection_thread.start()
    return "Sign detection started!"
```

- Starts the sign detection pipeline
- Runs in a background thread to keep Flask responsive
- Calls the core detection logic implemented in model_core

### Fetch Latest Output

```python
@app.route('/get-latest', methods=['GET'])
def get_latest():
    return model_core.get_latest_sentence()
```

Returns the latest detected sentence as plain text.

```python
@app.route('/get-latest-hindi', methods=['GET'])
def get_latest_hindi():
    return model_core.get_latest_hindi()
```

Returns the latest sentence translated into Hindi (if supported by `model_core`).

### Clearing Detected Output

```python
@app.route('/clear-last-word', methods=['POST'])
def clear_last_word():
    model_core.clear_last_word()
```

Removes the most recently detected word.

```python
@app.route('/clear-all', methods=['POST'])
def clear_all():
    model_core.clear_all()
```

Clears all detected words and resets the sentence buffer.

These endpoints allow UI controls such as *Undo* or *Reset*.

## Text-to-Speech Playback

### Asynchronous Speech Function

```python
def play_speech_async(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_path = fp.name

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.remove(temp_path)
    except Exception as e:
        print(f"Speech playback error: {e}")
```

This function:

- Converts text to speech using **Google Text-to-Speech**
- Saves audio to a temporary `.mp3` file
- Plays audio using `pygame`
- Deletes the temporary file after playback
- Runs in a separate thread to avoid blocking the server

Error handling ensures the app continues running even if playback fails.

### Speak Detected Sentence

#### `speak_sentence()`

```python
@app.route('/speak-sentence', methods=['POST'])
def speak_sentence():
    sentence = model_core.get_latest_sentence()
    if not sentence:
        return "No sentence available to speak."

    # Run speech playback asynchronously so Flask responds immediately
    Thread(target=play_speech_async, args=(sentence,), daemon=True).start()
    return "Speech playback started."
```

- Fetches the latest detected sentence
- Starts speech playback asynchronously
- Returns immediately to the client

This allows speech output without freezing the UI or server.

## Threading Strategy

This script uses threads to:

- Run sign detection independently of Flask
- Play audio without blocking requests

This ensures:

- Smooth UI interaction
- Continuous real-time detection
- Responsive web server behavior

## Application Startup

```python
if __name__ == '__main__':
    app.run(debug=True)
```

- Starts the Flask development server
- `debug=True` enables live reload and error messages (should be disabled in production)

## Summary

`app.py`:

- Acts as the **bridge between UI, detection logic, and speech output**
- Keeps sign detection and speech playback non-blocking
- Exposes simple API endpoints for frontend interaction
- Enables real-time sign-to-speech functionality

It forms the **final user-facing layer** of the Sign Language to Speech Conversion project.