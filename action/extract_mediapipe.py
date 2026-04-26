import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import savgol_filter

VIDEO_ROOTS = [
    '/mnt/d/dataset/ETRI-LivingLab/RGB(P01-P20)/',
    '/mnt/d/dataset/ETRI-LivingLab/RGB(P201-P230)/'
]
OUTPUT_ROOT = '/mnt/d/numpy_out_v6/'
CLASSES     = ["A001", "A016", "A024", "A039", "A053", "A054"]
SEQ_LEN     = 60
STRIDE      = 20

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for class_name in CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, class_name), exist_ok=True)

# =====================================================================
# 피처 추출 함수 (114차원)
# =====================================================================
def extract_features_114(seq_60_33_3):
    seq = seq_60_33_3
    T = seq.shape[0]

    # 1. 기본 좌표 (99)
    flat = seq.reshape(T, 99)

    # 랜드마크 슬라이싱 (x, y만)
    nose       = seq[:, 0,  :2]
    l_eye      = seq[:, 2,  :2]
    r_eye      = seq[:, 5,  :2]
    l_ear      = seq[:, 7,  :2]
    r_ear      = seq[:, 8,  :2]
    l_wrist    = seq[:, 15, :2]
    r_wrist    = seq[:, 16, :2]
    l_shoulder = seq[:, 11, :2]
    r_shoulder = seq[:, 12, :2]
    l_elbow    = seq[:, 13, :2]
    r_elbow    = seq[:, 14, :2]
    l_hip      = seq[:, 23, :2]
    r_hip      = seq[:, 24, :2]
    l_knee     = seq[:, 25, :2]
    r_knee     = seq[:, 26, :2]

    # 2. 거리 피처 (4)
    d_wrist_nose  = (np.linalg.norm(l_wrist - nose,  axis=1, keepdims=True) +
                     np.linalg.norm(r_wrist - nose,  axis=1, keepdims=True)) / 2.0
    d_wrist_ear   = (np.linalg.norm(l_wrist - l_ear, axis=1, keepdims=True) +
                     np.linalg.norm(r_wrist - r_ear, axis=1, keepdims=True)) / 2.0
    d_wrist_eye   = (np.linalg.norm(l_wrist - l_eye, axis=1, keepdims=True) +
                     np.linalg.norm(r_wrist - r_eye, axis=1, keepdims=True)) / 2.0
    d_wrist_wrist = np.linalg.norm(l_wrist - r_wrist, axis=1, keepdims=True)

    # 3. 관절 각도 (2)
    def calc_angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.sum(ba * bc, axis=1) / (
            np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
        return np.arccos(np.clip(cos, -1, 1)).reshape(-1, 1)

    angle_elbow = (calc_angle(l_shoulder, l_elbow, l_wrist) +
                   calc_angle(r_shoulder, r_elbow, r_wrist)) / 2.0
    angle_torso  = calc_angle(
        (l_shoulder + r_shoulder) / 2.0,
        (l_hip + r_hip) / 2.0,
        (l_knee + r_knee) / 2.0
    )

    # 4. Motion Energy 상/하체 분리 (2)
    motion_upper = np.std(seq[:, :23, :2], axis=(1, 2)).reshape(T, 1)
    motion_lower = np.std(seq[:, 23:, :2], axis=(1, 2)).reshape(T, 1)

    # 5. Bounding Box Ratio (1)
    width      = np.max(seq[:, :, 0], axis=1) - np.min(seq[:, :, 0], axis=1)
    height     = np.max(seq[:, :, 1], axis=1) - np.min(seq[:, :, 1], axis=1)
    bbox_ratio = (height / (width + 1e-8)).reshape(T, 1)

    # 6. 수직 이동량 (2) - 전체 프레임에 동일값 브로드캐스트
    dy_hip_val = ((l_hip[-1, 1] + r_hip[-1, 1]) / 2.0 -
                  (l_hip[0,  1] + r_hip[0,  1]) / 2.0)
    dy_sh_val  = ((l_shoulder[-1, 1] + r_shoulder[-1, 1]) / 2.0 -
                  (l_shoulder[0,  1] + r_shoulder[0,  1]) / 2.0)
    dy_hip = np.full((T, 1), dy_hip_val, dtype=np.float32)
    dy_sh  = np.full((T, 1), dy_sh_val,  dtype=np.float32)

    # 7. 손목 속도 X,Y (4) - Z 제외
    v_l = np.diff(l_wrist, axis=0, prepend=l_wrist[:1, :])  # (T, 2)
    v_r = np.diff(r_wrist, axis=0, prepend=r_wrist[:1, :])  # (T, 2)

    return np.hstack([
        flat,           # 99
        d_wrist_nose,   # 1
        d_wrist_ear,    # 1
        d_wrist_eye,    # 1
        d_wrist_wrist,  # 1
        angle_elbow,    # 1
        angle_torso,    # 1
        motion_upper,   # 1
        motion_lower,   # 1
        bbox_ratio,     # 1
        dy_hip,         # 1
        dy_sh,          # 1
        v_l,            # 2
        v_r,            # 2
    ]).astype(np.float32)  # (T, 114)


# =====================================================================
# 메인 추출 루프
# =====================================================================
total_saved = 0

for video_root in VIDEO_ROOTS:
    if not os.path.exists(video_root): continue
    for subject in sorted(os.listdir(video_root)):
        subject_path = os.path.join(video_root, subject)
        if not os.path.isdir(subject_path): continue
        for fname in sorted(os.listdir(subject_path)):
            if not fname.endswith('.mp4'): continue
            class_name = fname[:4]
            if class_name not in CLASSES: continue

            cap = cv2.VideoCapture(os.path.join(subject_path, fname))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    frames.append(np.array([[lm.x, lm.y, lm.z]
                                            for lm in res.pose_landmarks.landmark]))
                elif frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((33, 3)))
            cap.release()

            if len(frames) < SEQ_LEN: continue

            frames_np = np.array(frames, dtype=np.float32)
            T = frames_np.shape[0]

            # Savitzky-Golay 필터
            win_len = min(9, T if T % 2 == 1 else T - 1)
            if win_len < 5:
                continue
            for j in range(33):
                for k in range(3):
                    frames_np[:, j, k] = savgol_filter(
                        frames_np[:, j, k], win_len, 3)

            # 슬라이딩 윈도우
            window_idx = 0
            for start in range(0, T - SEQ_LEN + 1, STRIDE):
                seq = frames_np[start:start + SEQ_LEN].copy()

                # 힙 중심 이동 (X, Y만)
                hip_mid = (seq[:, 23:24, :2] + seq[:, 24:25, :2]) / 2.0
                seq[:, :, :2] -= hip_mid

                # 몸통 길이 스케일링 (X, Y만, Z 제외)
                shoulder_mid  = (seq[:, 11, :2] + seq[:, 12, :2]) / 2.0
                hip_mid_2d    = (seq[:, 23, :2] + seq[:, 24, :2]) / 2.0
                torso_len     = np.mean(np.linalg.norm(shoulder_mid - hip_mid_2d, axis=1))
                if torso_len < 0.02:
                    torso_len = 0.02
                seq[:, :, :2] /= torso_len

                # 피처 추출 → (60, 114)
                features = extract_features_114(seq)

                out_name = fname.replace('.mp4', f'_w{window_idx:03d}.npy')
                out_path = os.path.join(OUTPUT_ROOT, class_name, out_name)
                np.save(out_path, features)
                window_idx += 1

            total_saved += window_idx
            print(f"[{class_name}] {fname} -> {window_idx}개 생성")

print(f"\n추출 완료. 총 {total_saved}개 저장됨.")