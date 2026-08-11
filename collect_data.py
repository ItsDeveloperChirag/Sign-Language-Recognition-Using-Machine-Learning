import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Collect hand landmark data for sign language.')
parser.add_argument('--labels', nargs='+', default=['A','B'], help='Labels to record (e.g. A B C ...)')
parser.add_argument('--samples', type=int, default=100, help='Number of samples per label')
parser.add_argument('--output', type=str, default='./data', help='Output data directory')
parser.add_argument('--camera', type=int, default=0, help='Camera device index')
args = parser.parse_args()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = Path(args.output)
DATA_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f'Cannot open camera {args.camera}')

print("Instructions:")
print(" - Make sure your hand is visible to the camera.")
print(" - For each label, press 's' to start capturing samples.")
print(" - Press 'q' at any time to quit.")
print()

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    for label in args.labels:
        label_dir = DATA_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f'Ready to collect for label: {label}. {args.samples} samples needed.')
        print('Position your hand and press "s" to start...')
        # wait for 's'
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, f'Label: {label} - press s to start', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Collect - preview', display)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                break
            if k == ord('q'):
                print('Quitting collection.')
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

        count = 0
        while count < args.samples:
            ret, frame = cap.read()
            if not ret:
                continue
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if res.multi_hand_landmarks:
                # take first detected hand
                hand_landmarks = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                # save numpy array
                np.save(label_dir / f'{count:04d}.npy', np.array(landmarks, dtype=np.float32))
                count += 1
                cv2.putText(img_bgr, f'{label} samples: {count}/{args.samples}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(img_bgr, 'No hand detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow('Collect', img_bgr)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print('Quitting collection.')
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

print('Data collection complete.')
cap.release()
cv2.destroyAllWindows()
