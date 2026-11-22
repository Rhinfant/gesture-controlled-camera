# Gesture-Controlled Smart Camera

Control real hardware with hand gestures. A camera feed is processed with MediaPipe to extract 21 hand keypoints, and a **custom PyTorch classifier trained on my own dataset** recognizes gestures in real time. Recognized gestures will drive an Arduino (relay, servo, LCD) and an ESP32-CAM.

## Pipeline
```
camera -> MediaPipe hand landmarks (21 x 3) -> normalization -> MLP classifier -> temporal smoothing -> command
```
- **Normalization:** keypoints are translated to the wrist, scaled, and left hands are mirrored, so the model is independent of hand position, distance and handedness.
- **Classifier:** 63 -> 128 -> 64 -> N MLP with dropout, trained with Gaussian-noise augmentation.
- **Smoothing:** a gesture is accepted only if it wins the majority of the last 7 frames with ≥80% confidence, so single-frame errors don't trigger hardware.

## Gestures
`yumruk` (fist) · `avuc` (open palm) · `iki_parmak` (two fingers) · `bas_parmak_yukari` (thumbs up) · `isaret_parmagi` (index finger)

## Usage
```bash
pip install -r requirements.txt
python collect.py   # press 1-5 to record 200 samples of each gesture
python train.py     # trains and prints accuracy + confusion matrix
python live.py      # real-time recognition
```

## Roadmap
- [x] Data collection, training and live recognition with a laptop webcam
- [ ] Send commands to Arduino over serial (relay, servo, LCD)
- [ ] Use the ESP32-CAM as the camera
- [ ] Face tracking with a servo-mounted camera (PID control)
