"""
Canlı hareket tanıma (+ isteğe bağlı Arduino kontrolü).

    python live.py                 # sadece ekranda göster
    python live.py --port COM3     # tanınan hareketleri Arduino'ya da gönder

Bir hareket ancak son 7 karenin çoğunluğunda ve %90+ güvenle görülürse "kabul" edilir.
Bu, tek karelik yanlış tahminlerin röleyi/servoyu boşuna tetiklemesini engeller.
"""

import argparse
import time
from collections import Counter, deque

import cv2
import torch

from common import create_landmarker, detect_hand, load_classifier, normalize

WINDOW = 7
MIN_CONF = 0.9

# Hareket -> Arduino'ya gidecek tek karakterlik komut
COMMANDS = {
    "avuc": "A",
    "yumruk": "Y",
    "isaret_parmagi": "I",
    "iki_parmak": "V",
    "bas_parmak_yukari": "B",
    None: "N",
}


def open_arduino(port):
    import serial
    ser = serial.Serial(port, 115200, timeout=0)
    time.sleep(2)  # Port açılınca Uno kendini resetler, açılmasını bekle
    print(f"Arduino bağlandı: {port}")
    return ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Arduino'nun seri portu, örn. COM3")
    args = ap.parse_args()

    arduino = open_arduino(args.port) if args.port else None
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
            print("Hareket:", current)
            if arduino:
                arduino.write(COMMANDS[current].encode())  # sadece değişince gönder

        cv2.putText(frame, f"{label or '-'} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Kabul: {current or '-'}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Canli tanima", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if arduino:
        arduino.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
