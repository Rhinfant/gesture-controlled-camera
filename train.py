"""
Toplanan veriyle hareket sınıflandırıcısını eğitir.

    python train.py

Test verisi OTURUMA göre ayrılır: her hareketin son kaydı tamamen teste gider ve model onu hiç görmez.
Rastgele bölme yapılsaydı aynı kaydın neredeyse aynı kareleri hem eğitime hem teste düşer,
doğruluk yapay olarak yüksek çıkardı (veri sızıntısı).
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import CLASSIFIER_PATH, DATA_PATH, GESTURES, GestureNet

EPOCHS = 60
BATCH = 64
SEED = 0


def session_split(df):
    """Her hareketin en son oturumunu test için ayırır (en az 2 oturumu olan hareketler için)."""
    test_sessions = []
    for label, group in df.groupby("label"):
        sessions = sorted(group["session"].unique())
        if len(sessions) < 2:
            print(f"UYARI: '{label}' için tek kayıt var, test edilemiyor. En az 2 kez kaydet.")
            continue
        test_sessions.append(sessions[-1])
    is_test = df["session"].isin(test_sessions)
    return df[~is_test], df[is_test]


def to_tensors(df):
    X = torch.from_numpy(df.drop(columns=["label", "session"]).to_numpy(np.float32))
    y = torch.from_numpy(df["label"].map(GESTURES.index).to_numpy(np.int64))
    return X, y


def main():
    torch.manual_seed(SEED)

    df = pd.read_csv(DATA_PATH)
    if "session" not in df.columns:
        raise SystemExit("data.csv eski formatta. Dosyayı silip collect.py ile yeniden kayıt yap.")
    df = df[df["label"].isin(GESTURES)]
    print("Sınıf başına örnek:\n", df["label"].value_counts().to_string(), "\n")

    train_df, test_df = session_split(df)
    X_tr, y_tr = to_tensors(train_df)
    X_te, y_te = to_tensors(test_df)
    print(f"Eğitim: {len(X_tr)} örnek, test: {len(X_te)} örnek (hiç görülmemiş kayıtlar)\n")

    model = GestureNet(len(GESTURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            # Veri artırma: küçük rastgele gürültü, modeli titreyen ele karşı dayanıklı yapar
            xb = X_tr[b] + 0.02 * torch.randn_like(X_tr[b])
            loss = loss_fn(model(xb), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 10 == 0 and len(X_te):
            model.eval()
            with torch.no_grad():
                acc = (model(X_te).argmax(1) == y_te).float().mean().item()
            print(f"epoch {epoch:3d}  loss {loss.item():.3f}  test doğruluğu {acc:.3f}")

    if len(X_te):
        model.eval()
        with torch.no_grad():
            pred = model(X_te).argmax(1).numpy()
        cm = pd.crosstab(pd.Series([GESTURES[i] for i in y_te.numpy()], name="gerçek"),
                         pd.Series([GESTURES[i] for i in pred], name="tahmin"))
        print("\nKarışıklık matrisi (satır = gerçek, sütun = tahmin):\n", cm.to_string())

    torch.save({"state_dict": model.state_dict(), "gestures": GESTURES}, CLASSIFIER_PATH)
    print(f"\nModel kaydedildi: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
