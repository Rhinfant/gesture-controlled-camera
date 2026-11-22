# Gesture-Controlled Smart Camera

Control real hardware with hand gestures. A camera feed is processed with MediaPipe to extract 21 hand keypoints, and a **custom PyTorch classifier trained on my own dataset** recognizes gestures in real time. Recognized gestures are sent to an Arduino that drives a relay, a servo and an LCD.

## Pipeline
```
camera -> MediaPipe hand landmarks (21 x 3) -> normalization -> MLP classifier -> temporal smoothing -> serial command -> Arduino
```
- **Normalization:** keypoints are translated to the wrist, scaled, and left hands are mirrored, so the model is independent of hand position, distance and handedness.
- **Classifier:** 63 -> 128 -> 64 -> N MLP with dropout, trained with Gaussian-noise augmentation.
- **"None" class:** a dedicated `yok` class (back of the hand, half-open hands, transitions) so the model doesn't force every pose into a gesture.
- **Session-based evaluation:** the last recording of each gesture is held out for testing. A random frame split leaked near-identical frames into the test set and inflated accuracy to 99.9%.
- **Smoothing:** a gesture is accepted only if it wins the majority of the last 7 frames with ≥90% confidence, so single-frame errors don't trigger hardware.

## Gestures and actions
| Gesture | Command | Action |
|---|---|---|
| `avuc` (open palm) | `A` | Relay ON |
| `yumruk` (fist) | `Y` | Relay OFF |
| `isaret_parmagi` (index finger) | `I` | Servo 30° |
| `iki_parmak` (two fingers) | `V` | Servo 90° |
| `bas_parmak_yukari` (thumbs up) | `B` | Servo 150° |
| `yok` (none) | `N` | No action |

## Results
~15,000 samples collected with a laptop webcam. Accuracy on held-out recording sessions: **~99%**.
Limitation: all data was recorded by one person in one environment; accuracy on other users and lighting conditions has not been measured yet.

## Hardware
Arduino Uno, SG90 servo, 5V relay module, 16x2 LCD, 10K potentiometer (LCD contrast).

| Part | Arduino pin |
|---|---|
| Servo signal | D9 |
| Relay signal (S/IN) | D7 |
| LCD RS, E, D4, D5, D6, D7 | D12, D11, D5, D4, D3, D2 |

## Usage
```bash
pip install -r requirements.txt
python collect.py              # press 0-5 to record 200 samples of each gesture
python train.py                # trains and prints accuracy + confusion matrix
python live.py                 # real-time recognition
python live.py --port COM3     # real-time recognition + Arduino control
```
Upload `arduino/gesture_controller/gesture_controller.ino` to the Arduino first.

## Roadmap
- [x] Data collection, training and live recognition with a laptop webcam
- [x] Send commands to Arduino over serial (relay, servo, LCD)
- [ ] Dynamic gestures (swipes, circles) with a sequence model (LSTM / 1D-CNN)
- [ ] Use the ESP32-CAM as the camera
- [ ] Face tracking with a servo-mounted camera (PID control)
