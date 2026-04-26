"""
train_final.py
═══════════════════════════════════════════════════════════════════════════════
최종 모델 학습: V6 MultiScale+GRU
Ablation 12종 실험 결과 최고 성능 모델 (Val F1: 0.9785)

■ 실행 환경
  - Google Colab (GPU 권장)
  - 또는 로컬 Python 3.10+

■ 데이터 경로 설정
  NUMPY_ROOT: 전처리된 .npy 파일 폴더 (extract_mediapipe3.py로 생성)
  MODEL_SAVE: 학습된 모델 저장 경로

■ 클래스 (4개)
  A016: 머리 빗기  → Proactive Talk 트리거
  A039: 박수 치기  → Proactive Talk 트리거
  A053: 쓰러지기   → 낙상 경보
  A054: 누워있다 일어나기 → Proactive Talk 트리거

■ 모델 선정 근거
  Reference: InceptionTime (Fawaz et al., 2020)
  다중 커널(3/5/7) 병렬 CNN이 단일 커널 대비 시계열 분류에서 우수함
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Colab 전용: Drive 마운트 ───────────────────────────────────────────────
# 로컬 실행 시 아래 3줄 주석 처리
# from google.colab import drive
# drive.mount('/content/drive')
# !pip install torchinfo -q

import os, shutil, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 경로 및 하이퍼파라미터 설정
# ─────────────────────────────────────────────────────────────────────────────
# ── Colab 경로
# NUMPY_ROOT = "/content/numpy_out_v6"
# MODEL_SAVE = "/content/drive/MyDrive/models/V6_MultiScale_GRU.pth"

# ── 로컬 경로 (본인 환경에 맞게 수정)
NUMPY_ROOT = "./numpy_out_v6"
MODEL_SAVE = "./models/V6_MultiScale_GRU.pth"

CLASSES       = ["A016", "A039", "A053", "A054"]
NUM_CLASSES   = 4
SEQ_LEN       = 60
INPUT_SIZE    = 114
EPOCHS        = 150
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
PATIENCE      = 20

os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 데이터 로딩 (영상 단위 분할로 data leakage 방지)
# ─────────────────────────────────────────────────────────────────────────────
print("\n데이터 로딩 중...")

video_groups = {}
for label_idx, class_name in enumerate(CLASSES):
    folder = os.path.join(NUMPY_ROOT, class_name)
    if not os.path.exists(folder):
        print(f"  ⚠️  {class_name} 폴더 없음")
        continue
    for fname in sorted(f for f in os.listdir(folder) if f.endswith('.npy')):
        video_key = fname.rsplit('_w', 1)[0]
        key = (class_name, video_key, label_idx)
        video_groups.setdefault(key, []).append(os.path.join(folder, fname))

group_keys = list(video_groups.keys())
train_groups, val_groups = train_test_split(
    group_keys, test_size=0.2, random_state=42,
    stratify=[k[0] for k in group_keys]
)

def load_windows(groups):
    X, y = [], []
    for key in groups:
        for fpath in video_groups[key]:
            data = np.load(fpath)
            if data.shape == (SEQ_LEN, INPUT_SIZE):
                X.append(data)
                y.append(key[2])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X_train_raw, y_train_raw = load_windows(train_groups)
X_val,       y_val       = load_windows(val_groups)

print(f"학습: {len(X_train_raw)}개 | 검증: {len(X_val)}개")
print(f"클래스별 학습: {[int(sum(y_train_raw==i)) for i in range(NUM_CLASSES)]}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 데이터 증강
# ─────────────────────────────────────────────────────────────────────────────
FLIP_PAIRS = [(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),
              (23,24),(25,26),(27,28),(29,30),(31,32)]

def time_warp(seq, sigma=0.2):
    T  = seq.shape[0]
    tt = np.arange(T, dtype=np.float32)
    tt_new = np.clip(tt + np.random.normal(0, sigma*T, T).cumsum(), 0, T-1)
    return np.array([np.interp(tt, tt_new, seq[:, d]) for d in range(seq.shape[1])],
                    dtype=np.float32).T

def cutout_time(seq, n_holes=2, hole_len=6):
    result = seq.copy()
    for _ in range(n_holes):
        start = np.random.randint(0, seq.shape[0] - hole_len)
        result[start:start+hole_len] = 0.0
    return result.astype(np.float32)

def augment(seq, label=None):
    results = [seq]

    # 좌우 반전
    s = seq[:, :99].reshape(SEQ_LEN, 33, 3).copy()
    s[:, :, 0] *= -1
    for i, j in FLIP_PAIRS:
        s[:, [i, j]] = s[:, [j, i]]
    rest = seq[:, 99:].copy()
    results.append(np.hstack([s.reshape(SEQ_LEN, 99), rest]).astype(np.float32))

    # 가우시안 노이즈
    noisy = seq.copy()
    noisy[:, :99] += np.random.normal(0, 0.005, (SEQ_LEN, 99)).astype(np.float32)
    results.append(noisy)

    results.append(time_warp(seq))
    results.append(cutout_time(seq))

    # A053/A054 추가 증강 (낙상 관련)
    if label in (2, 3):
        n2 = seq.copy(); n2[:, :99] += np.random.normal(0, 0.01, (SEQ_LEN, 99)).astype(np.float32)
        sc = seq.copy(); sc[:, :99] *= np.random.uniform(0.95, 1.05)
        fast = np.repeat(seq[::2], 2, axis=0)[:SEQ_LEN]
        slow = np.repeat(seq, 2, axis=0)[:SEQ_LEN]
        results.extend([n2, sc.astype(np.float32), fast.astype(np.float32), slow.astype(np.float32)])

    return results

print("\n증강 중...")
X_aug, y_aug = [], []
for x, y in zip(X_train_raw, y_train_raw):
    for a in augment(x, int(y)):
        X_aug.append(a); y_aug.append(y)
X_train = np.array(X_aug, dtype=np.float32)
y_train = np.array(y_aug, dtype=np.int64)
print(f"증강 후: {len(X_train)}개")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 정규화
# ─────────────────────────────────────────────────────────────────────────────
X_mean = X_train.mean(axis=(0,1), keepdims=True)
X_std  = X_train.std(axis=(0,1),  keepdims=True) + 1e-8
X_train_norm = (X_train - X_mean) / X_std
X_val_norm   = (X_val   - X_mean) / X_std


# ─────────────────────────────────────────────────────────────────────────────
# 5. 클래스 가중치 및 DataLoader
# ─────────────────────────────────────────────────────────────────────────────
class_counts  = np.array([sum(y_train==i) for i in range(NUM_CLASSES)])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"클래스 가중치: {[f'{w:.3f}' for w in class_weights]}")

class ActionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(ActionDataset(X_train_norm, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(ActionDataset(X_val_norm, y_val),
                          batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 모델 정의: V6 MultiScale+GRU
#    Reference: InceptionTime (Fawaz et al., 2020)
#    커널 3/5/7 병렬 → concat(128ch) → MaxPool → GRU → FC
# ─────────────────────────────────────────────────────────────────────────────
class ModelV6_MultiScale_GRU(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv3   = nn.Conv1d(input_size, 48, kernel_size=3, padding=1)
        self.conv5   = nn.Conv1d(input_size, 48, kernel_size=5, padding=2)
        self.conv7   = nn.Conv1d(input_size, 32, kernel_size=7, padding=3)
        self.bn      = nn.BatchNorm1d(128)   # 48+48+32=128
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(2)
        self.gru     = nn.GRU(128, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)                        # (B, F, T)
        x = self.relu(self.bn(torch.cat([
            self.conv3(x), self.conv5(x), self.conv7(x)
        ], dim=1)))                                   # (B, 128, T)
        x = self.pool(x)                              # (B, 128, T/2)
        x = x.transpose(1, 2)                        # (B, T/2, 128)
        x, _ = self.gru(x)
        return self.fc(self.dropout(x[:, -1, :]))

model = ModelV6_MultiScale_GRU().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n모델 파라미터: {n_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MixUp
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(x, y, num_classes, alpha=0.1):
    lam     = np.random.beta(alpha, alpha)
    idx     = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1-lam) * x[idx]
    mixed_y = lam * F.one_hot(y, num_classes).float() + \
              (1-lam) * F.one_hot(y[idx], num_classes).float()
    return mixed_x, mixed_y


# ─────────────────────────────────────────────────────────────────────────────
# 8. 학습
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-5
)

best_val_acc = 0.0
best_val_f1  = 0.0
best_state   = None
patience_cnt = 0
history      = {"train_acc": [], "val_acc": [], "val_f1": []}
t_start      = time.time()

print("\n학습 시작...")
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        X_mix, y_mix = mixup_batch(X_b, y_b, NUM_CLASSES)
        loss = -(y_mix * F.log_softmax(model(X_mix), dim=1)).sum(1).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            preds = model(X_b).argmax(1)
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)

    scheduler.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            preds = model(X_b.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y_b.numpy())

    train_acc = correct / total
    val_acc   = accuracy_score(all_labels, all_preds)
    val_f1    = f1_score(all_labels, all_preds, average='macro')
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    if (epoch+1) % 10 == 0:
        print(f"Ep {epoch+1:3d} | train: {train_acc:.3f} | "
              f"val: {val_acc:.3f} | F1: {val_f1:.3f} | "
              f"gap: {train_acc-val_acc:+.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1  = val_f1
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
        torch.save({
            'model_state': best_state,
            'X_mean': X_mean, 'X_std': X_std,
            'classes': CLASSES,
            'arch': 'V6_MultiScale_GRU',
        }, MODEL_SAVE)
        print(f"  ✅ 저장 (val_acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping @ epoch {epoch+1}")
            break

elapsed = time.time() - t_start
print(f"\n학습 완료: {elapsed:.0f}s")
print(f"최고 Val Acc: {best_val_acc:.4f} | Val F1: {best_val_f1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 최종 평가
# ─────────────────────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in val_loader:
        preds = model(X_b.to(device)).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(y_b.numpy())

print("\n" + "="*60)
print("Classification Report")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — V6: MultiScale+GRU")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()*0.5 else 'black',
                fontsize=13, fontweight='bold')
plt.colorbar(im); plt.tight_layout()
plt.savefig(MODEL_SAVE.replace('.pth', '_cm.png'), dpi=150)
plt.show()

# 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, len(history["val_acc"]) + 1)
axes[0].plot(ep, history["train_acc"], label="Train"); axes[0].plot(ep, history["val_acc"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ep, history["val_f1"], label="Val F1", color='green')
axes[1].set_title("Val F1 (macro)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle("V6: MultiScale+GRU Training Curve")
plt.tight_layout()
plt.savefig(MODEL_SAVE.replace('.pth', '_curve.png'), dpi=150)
plt.show()
