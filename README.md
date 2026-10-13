# SignLanguageRecognition - Ready to Run (Alphabet recognizer)

## What's included
- `collect_data.py`  : collect hand landmarks (MediaPipe) for each label (e.g., A, B, C)
- `train_model.py`   : train a simple neural network (MLP) on saved landmarks
- `main.py`          : run real-time predictions using your webcam
- `data/`            : where collected `.npy` files are saved (one folder per label)
- `model/`           : trained model will be saved here
- `requirements.txt` : Python packages used

## Quick start
1. (Optional) create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate   # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Collect data (example for labels A and B):
   ```bash
   python collect_data.py --labels A B C D --samples 120
   ```
   For each label: position your hand, press `s` to start, and the script will save samples.
4. Train:
   ```bash
   python train_model.py --data ./data --output ./model/sign_model.h5 --epochs 60
   ```
5. Run real-time demo:
   ```bash
   python main.py --model ./model/sign_model.h5
   ```
6. Press `q` in any window to quit.

## Notes & tips
- Use good lighting and varied hand positions for robust models.
- Start with 5-10 labels and 100+ samples per label to get decent performance.
- For dynamic gestures (moving hands), you'll need to collect sequences and train an LSTM or temporal model (this project is for static signs/alphabet).
