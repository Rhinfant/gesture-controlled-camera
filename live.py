"""
Canlı hareket tanıma.

    python live.py

Bir hareket ancak son 7 karenin çoğunluğunda ve %80+ güvenle görülürse "kabul" edilir.
Bu, tek karelik yanlış tahminlerin röleyi/servoyu boşuna tetiklemesini engeller.
"""

import time
from collections import Counter, deque

import cv2
import torch

from common import create_landmarker, detect_hand, load_classifier, normalize

WINDOW = 7
MIN_CONF = 0.9


def main():
    landmarker = create_landmarker()
    model, gestures = load_classifier()
    cap = cv2.VideoCapture(0)
    start = time.time()
    history = deque(maxlen=WINDOW)
    current = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        points, hand = detect_hand(landmarker, frame, int((time.time() - start) * 1000))

        label, conf = None, 0.0
        if points is not None:
            with torch.no_grad():
                probs = torch.softmax(model(torch.from_numpy(normalize(points, hand)).unsqueeze(0)), 1)[0]
            conf, i = probs.max(0)
            conf = conf.item()
            label = gestures[i] if conf >= MIN_CONF and gestures[i] != "yok" else None
        history.append(label)

        # Kararlı hareket: pencerenin çoğunluğu aynı etiketse
        top, n = Counter(history).most_common(1)[0]
        stable = top if top is not None and n > WINDOW // 2 else None
        if stable != current:
            current = stable
            print("Hareket:", current)  # 2. adımda burası Arduino'ya komut gönderecek

        cv2.putText(frame, f"{label or '-'} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Kabul: {current or '-'}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Canli tanima", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
