Virtual Air Drawing with OCR
Draw letters and numbers **in the air using your index finger** via webcam.
The app tracks your hand using MediaPipe, lets you draw on a virtual canvas,
and recognizes handwritten characters (A-Z, a-z, 0-9) using Tesseract OCR.

Features
Real-time hand tracking using MediaPipe Hand Landmarker
Draw on a virtual canvas using your index finger
Recognizes uppercase, lowercase letters and digits (A-Z, a-z, 0-9)
Multi-PSM OCR voting for better accuracy
Live debug window showing what Tesseract sees
No mouse or keyboard needed to draw

Demo
Draw in the air -> Press R -> See recognized character

Requirements
Python 3.9 -> 3.12
Webcam
Tesseract OCR v5(https://github.com/UB-Mannheim/tesseract/wiki) installed on your system

Python dependencies
Install with:
bash
pip install mediapipe opencv-python pytesseract pyspellchecker

How It Works
1. Webcam feed is processed by **MediaPipe Hand Landmarker** to detect the index finger tip
2. When drawing is ON, the finger tip position is traced onto a virtual canvas
3. On pressing R, the canvas is **cropped, upscaled, and thresholded**
4. **Tesseract OCR** runs with multiple PSM modes (10, 8, 7, 13, 6) on both polarities
5. The most common result across all runs is shown as the final answer

Controls
 Key  Action 
D - Toggle drawing ON / OFF 
C - Clear the canvas 
R - Recognize drawn text 
Q - Quit 

Project Structure
project.py            # Main application
README.md

Tech Stack
MediaPipe(https://developers.google.com/mediapipe) - Hand tracking
OpenCV(https://opencv.org/) - Camera feed & drawing
Tesseract OCR(https://github.com/tesseract-ocr/tesseract) - Text recognition
pyspellchecker(https://pypi.org/project/pyspellchecker/) - Spell correction
