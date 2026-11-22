"""
Veri toplama: kameraya hareketi yap, ilgili numara tuşuna bas, 200 örnek kaydedilir.

    python collect.py

Tuşlar: 0..5 = hareket kaydet, q = çık
Her tuşa basış ayrı bir "oturum" olarak kaydedilir; train.py test verisini oturuma göre ayırır.

İpuçları:
- Kayıt sırasında elini döndür, yaklaştır, uzaklaştır, iki elini de kullan.
- 0 (yok) için: elin arkası, yarım açık el, hareketler arası geçiş, rastgele pozlar.
"""

import csv
import time

import cv2

from common import DATA_PATH, GESTURES, create_landmarker, detect_hand, normalize

SAMPLES_PER_PRESS = 200


def main():
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)
    start = time.time()

    new_file = not DATA_PATH.exists()
    f = open(DATA_PATH, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label", "session"] + [f"f{i}" for i in range(63)])

    recording_label, session, remaining = None, None, 0
    counts = {g: 0 for g in GESTURES}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # ayna görüntü, kullanması daha doğal
        points, hand = detect_hand(landmarker, frame, int((time.time() - start) * 1000))

        if points is not None:
            for x, y, _ in points:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 3, (0, 255, 0), -1)
            if remaining > 0:
                writer.writerow([recording_label, session] + normalize(points, hand).tolist())
                counts[recording_label] += 1
                remaining -= 1

        status = f"KAYIT: {recording_label} ({remaining})" if remaining else "Hazir - 0..5 tusuna bas"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for i, g in enumerate(GESTURES):
            cv2.putText(frame, f"{i}: {g} [{counts[g]}]", (10, 60 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Veri toplama", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key < ord("0") + len(GESTURES) and remaining == 0:
            recording_label = GESTURES[key - ord("0")]
            session = int(time.time() * 1000)  # her basış = yeni oturum
            remaining = SAMPLES_PER_PRESS

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Bu oturumda kaydedilen:", counts)


if __name__ == "__main__":
    main()
