"""Tüm scriptlerin ortak kullandığı parçalar: hareket listesi, el modeli, normalizasyon, sinir ağı."""

import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Tanınacak hareketler. Klavyede 0, 1, 2... tuşları bu sıraya karşılık gelir.
# "yok" = tanımlı hareketlerden hiçbiri (geçişler, yarım hareketler, elin arkası vs.).
# Bu sınıf olmazsa model her şeyi zorla 5 hareketten birine benzetir.
GESTURES = ["yok", "yumruk", "avuc", "iki_parmak", "bas_parmak_yukari", "isaret_parmagi"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path("hand_landmarker.task")
DATA_PATH = Path("data.csv")
CLASSIFIER_PATH = Path("gesture_model.pt")


def create_landmarker():
    """MediaPipe el eklem modeli. İlk çalıştırmada modeli indirir."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODEL_PATH.exists():
        print("El modeli indiriliyor...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand(landmarker, frame_bgr, timestamp_ms):
    """Görüntüde el bulursa (21 nokta x 3) dizisini ve sağ/sol bilgisini döndürür, yoksa (None, None)."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    if not result.hand_landmarks:
        return None, None
    points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    handedness = result.handedness[0][0].category_name  # "Left" / "Right"
    return points, handedness


def normalize(points, handedness):
    """
    Eklem noktalarını eldeki konumdan, boyuttan ve hangi el olduğundan bağımsız hale getirir:
      1. Bileği (0. nokta) orijine taşı  -> elin ekrandaki yeri önemsizleşir
      2. En uzak noktaya göre ölçekle    -> kameraya yakınlık önemsizleşir
      3. Sol eli aynala                  -> iki el için tek model yeter
    Sonuç: 63 elemanlı vektör.
    """
    p = points - points[0]
    p = p / (np.abs(p).max() + 1e-6)
    if handedness == "Left":
        p[:, 0] = -p[:, 0]
    return p.flatten()


class GestureNet(nn.Module):
    """63 girişli küçük bir çok katmanlı algılayıcı (MLP)."""

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_classifier():
    ckpt = torch.load(CLASSIFIER_PATH, map_location="cpu")
    model = GestureNet(len(ckpt["gestures"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["gestures"]
