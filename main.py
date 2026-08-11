import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Run real-time sign recognition.')
parser.add_argument('--model', type=str, default='./model/sign_model.h5', help='Path to trained model')
parser.add_argument('--labels', nargs='+', default=None, help='List of labels in same order used during training')
parser.add_argument('--camera', type=int, default=0, help='Camera device index')
args = parser.parse_args()

model_path = Path(args.model)
if not model_path.exists():
    raise RuntimeError(f'Model not found at {model_path}. Train a model first.')

model = load_model(str(model_path))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f'Cannot open camera {args.camera}')

# If labels not provided, attempt to read from data folders in ../data
if args.labels is None:
    # Try to auto-detect labels from sibling data folder
    data_dir = Path('./data')
    if data_dir.exists():
        labels = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    else:
        labels = []
else:
    labels = args.labels

print('Labels:', labels)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if res.multi_hand_landmarks:
            hand_landmarks = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            x = np.array([landmarks], dtype=np.float32)
            preds = model.predict(x)
            idx = int(np.argmax(preds))
            prob = float(np.max(preds))
            label = labels[idx] if idx < len(labels) else str(idx)
            cv2.putText(img_bgr, f'{label} ({prob:.2f})', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(img_bgr, 'No hand detected', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Sign Recognition', img_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
