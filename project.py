import cv2
import numpy as np
import pytesseract
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import Counter
import urllib.request
import shutil
import os
import time

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~8 MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Downloaded!")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=mp_vision.RunningMode.VIDEO,
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]
INDEX_FINGER_TIP = 8

def draw_hand_landmarks(frame, landmarks, w, h):
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(landmarks[s].x * w), int(landmarks[s].y * h)),
                 (int(landmarks[e].x * w), int(landmarks[e].y * h)),
                 (0, 255, 0), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 0, 0), -1)

tp = shutil.which("tesseract")
pytesseract.pytesseract.tesseract_cmd = tp if tp else r'C:\Users\tejas\tesseract.exe'

WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

def run_ocr(draw_canvas):
    """
    draw_canvas: pure black canvas with WHITE strokes (grayscale or BGR).
    Tesseract works best with BLACK text on WHITE background, so we invert.
    """
    # Convert to grayscale if BGR
    if len(draw_canvas.shape) == 3:
        gray = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
    else:
        gray = draw_canvas.copy()


    if cv2.countNonZero(gray) < 100:
        show_result("Nothing drawn!", [])
        return

    coords = cv2.findNonZero(gray)
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = 50
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + bw + pad, gray.shape[1])
    y2 = min(y + bh + pad, gray.shape[0])
    cropped = gray[y1:y2, x1:x2]

    target_h = 400
    scale    = target_h / cropped.shape[0]
    target_w = max(int(cropped.shape[1] * scale), target_h)
    resized  = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    _, wb = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel     = np.ones((3, 3), np.uint8)
    wb_dilated = cv2.dilate(wb, kernel, iterations=2)

    bordered_wb = cv2.copyMakeBorder(wb_dilated, 80, 80, 80, 80,
                                     cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(bordered_wb)
    candidates = []
    configs = [
        f'--oem 3 --psm 10 -c tessedit_char_whitelist={WHITELIST}', 
        f'--oem 3 --psm 8  -c tessedit_char_whitelist={WHITELIST}',  
        f'--oem 3 --psm 7  -c tessedit_char_whitelist={WHITELIST}', 
        f'--oem 3 --psm 13 -c tessedit_char_whitelist={WHITELIST}',  
        f'--oem 3 --psm 6  -c tessedit_char_whitelist={WHITELIST}',  
    ]

    for cfg in configs:
        for img in [inv, bordered_wb]:        
            try:
                raw    = pytesseract.image_to_string(img, config=cfg).strip()
                result = ''.join(c for c in raw if c.isalnum())
                if result:
                    candidates.append(result)
                    print(f"  cfg='{cfg[-10:]}' → '{result}'")
            except Exception as ex:
                print(f"  OCR error: {ex}")

    if not candidates:
        show_result("Not Recognized", [], inv)
        return


    single = [c for c in candidates if len(c) == 1]
    pool   = single if single else candidates
    best   = Counter(pool).most_common(1)[0][0].upper()
    print(f"  ✔ Final: {best}")

    show_result(best, list(set(candidates)), inv)


def show_result(text, candidates, debug_img=None):
    """Display recognition result window."""
    img = np.ones((280, 520, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Recognized Text:", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


    font_scale = 3.0 if len(text) == 1 else 1.5
    cv2.putText(img, text, (15, 180),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 200), 4)


    cv2.putText(img, f"All guesses: {candidates[:10]}", (15, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    cv2.imshow("Text Recognition", img)


    if debug_img is not None:
        disp = cv2.resize(debug_img, (400, 400))
        disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp_bgr, "What Tesseract sees", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("OCR Input", disp_bgr)

cap            = cv2.VideoCapture(0)
ret, frame     = cap.read()
h, w           = (frame.shape[0], frame.shape[1]) if ret else (480, 640)

draw_canvas    = np.zeros((h, w), dtype=np.uint8)

drawing        = False
prev_x, prev_y = None, None
start_time     = time.time()

print("=" * 40)
print("  D  → Toggle Drawing ON / OFF")
print("  C  → Clear canvas")
print("  R  → Recognize drawn text")
print("  Q  → Quit")
print("=" * 40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if draw_canvas.shape != (h, w):
        draw_canvas = np.zeros((h, w), dtype=np.uint8)

    rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    results      = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            draw_hand_landmarks(frame, hand_lms, w, h)
            tip  = hand_lms[INDEX_FINGER_TIP]
            x, y = int(tip.x * w), int(tip.y * h)

            if drawing:
                if prev_x is not None and prev_y is not None:
                    
                    cv2.line(draw_canvas, (prev_x, prev_y), (x, y), 255, 12)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)
    else:
        prev_x, prev_y = None, None

    canvas_bgr = cv2.cvtColor(draw_canvas, cv2.COLOR_GRAY2BGR)
    combined   = cv2.addWeighted(frame, 0.6, canvas_bgr, 0.9, 0)

    status     = "Drawing: ON" if drawing else "Drawing: OFF"
    status_col = (0, 230, 0)   if drawing else (0, 0, 230)
    cv2.putText(combined, status, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_col, 2)
    cv2.putText(combined, "D=Draw  C=Clear  R=Recognize  Q=Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    cv2.imshow("Virtual Drawing", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        print("Quitting...")
        break
    elif key == ord('d'):
        drawing = not drawing
        print(f"Drawing {'ON' if drawing else 'OFF'}")
    elif key == ord('c'):
        draw_canvas[:] = 0
        print("Canvas cleared")
    elif key == ord('r'):
        run_ocr(draw_canvas)

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()