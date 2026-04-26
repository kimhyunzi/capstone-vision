"""
action_inference_node.py
═══════════════════════════════════════════════════════════════════════════════
실시간 행동인식 ROS2 노드 — V6: MultiScale+GRU (Val F1: 0.9785)

■ 전처리 파이프라인 (extract_mediapipe3.py 완전 동일)
  1) MediaPipe → (33, 3) raw xyz 수집 → deque에 저장
  2) 60프레임 모이면 preprocess_window() 호출:
       a) 힙 중심 정규화: seq[:,:,:2] -= hip_midpoint
       b) 토르소 길이 스케일링: seq[:,:,:2] /= mean_torso_len
       c) extract_features_114() → (60, 114)
  3) 학습 통계(X_mean, X_std)로 정규화
  4) V6 모델 추론 → softmax → Temporal Voting

■ 최적화 근거
  - static_image_mode=False: BlazePose tracking 모드 (Bazarevsky et al. 2020)
  - deque(maxlen=60): O(1) 슬라이딩 윈도우
  - torch.inference_mode(): PyTorch Performance Tuning Guide 권장

■ 퍼블리시 토픽
  /action_recognition     (std_msgs/String):  JSON 결과
  /action_recognition_raw (std_msgs/Int32):   클래스 인덱스 (integrator용)

■ 독립 실행 (ROS2 없이 성능 확인)
  python action_inference_node.py --test /path/to/V6.pth [camera_id]
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import collections
import threading

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 0. 공통 설정
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES = 4
SEQ_LEN     = 60
INPUT_SIZE  = 114
CLASSES     = ["A016", "A039", "A053", "A054"]
LABEL_MAP   = {
    "A016": "Hair Combing",
    "A039": "Clapping",
    "A053": "FALL !!!",
    "A054": "Getting Up",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. 모델 아키텍처 — V6: MultiScale+GRU
#    Reference: InceptionTime (Fawaz et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────
class ModelV6_MultiScale_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3   = nn.Conv1d(INPUT_SIZE, 48, kernel_size=3, padding=1)
        self.conv5   = nn.Conv1d(INPUT_SIZE, 48, kernel_size=5, padding=2)
        self.conv7   = nn.Conv1d(INPUT_SIZE, 32, kernel_size=7, padding=3)
        self.bn      = nn.BatchNorm1d(128)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(2)
        self.gru     = nn.GRU(128, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn(torch.cat([
            self.conv3(x), self.conv5(x), self.conv7(x)
        ], dim=1)))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return self.fc(self.dropout(x[:, -1, :]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. 전처리: extract_mediapipe3.py와 완전 동일
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_window(raw_window: np.ndarray) -> np.ndarray:
    """
    (60, 33, 3) raw xyz → (60, 114) 피처.
    extract_mediapipe3.py의 전처리 + extract_features_114와 동일.
    """
    seq = raw_window.copy()   # (60, 33, 3)
    T   = seq.shape[0]

    # ── 힙 중심 정규화 (X, Y만) ──────────────────────────────────
    hip_mid        = (seq[:, 23:24, :2] + seq[:, 24:25, :2]) / 2.0
    seq[:, :, :2] -= hip_mid

    # ── 토르소 길이 스케일링 (X, Y만) ────────────────────────────
    shoulder_mid = (seq[:, 11, :2] + seq[:, 12, :2]) / 2.0
    hip_mid_2d   = (seq[:, 23, :2] + seq[:, 24, :2]) / 2.0
    torso_len    = float(np.mean(np.linalg.norm(shoulder_mid - hip_mid_2d, axis=1)))
    if torso_len < 0.02:
        torso_len = 0.02
    seq[:, :, :2] /= torso_len

    # ── extract_features_114 ──────────────────────────────────────
    flat       = seq.reshape(T, 99)
    nose       = seq[:, 0,  :2]
    l_eye      = seq[:, 2,  :2]
    r_eye      = seq[:, 5,  :2]   # noqa (학습 코드와 동일하게 유지)
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

    # 거리 피처 (4)
    d_wrist_nose  = (np.linalg.norm(l_wrist - nose,  axis=1, keepdims=True) +
                     np.linalg.norm(r_wrist - nose,  axis=1, keepdims=True)) / 2.0
    d_wrist_ear   = (np.linalg.norm(l_wrist - l_ear, axis=1, keepdims=True) +
                     np.linalg.norm(r_wrist - r_ear, axis=1, keepdims=True)) / 2.0
    d_wrist_eye   = (np.linalg.norm(l_wrist - l_eye, axis=1, keepdims=True) +
                     np.linalg.norm(r_wrist - r_eye, axis=1, keepdims=True)) / 2.0
    d_wrist_wrist = np.linalg.norm(l_wrist - r_wrist, axis=1, keepdims=True)

    # 관절 각도 (2)
    def calc_angle(a, b, c):
        ba  = a - b
        bc  = c - b
        cos = np.sum(ba * bc, axis=1) / (
            np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
        return np.arccos(np.clip(cos, -1, 1)).reshape(-1, 1)

    angle_elbow = (calc_angle(l_shoulder, l_elbow, l_wrist) +
                   calc_angle(r_shoulder, r_elbow, r_wrist)) / 2.0
    angle_torso  = calc_angle(
        (l_shoulder + r_shoulder) / 2.0,
        (l_hip      + r_hip)      / 2.0,
        (l_knee     + r_knee)     / 2.0,
    )

    # Motion Energy 상/하체 (2)
    motion_upper = np.std(seq[:, :23, :2], axis=(1, 2)).reshape(T, 1)
    motion_lower = np.std(seq[:, 23:, :2], axis=(1, 2)).reshape(T, 1)

    # BBox Ratio (1)
    width      = np.max(seq[:, :, 0], axis=1) - np.min(seq[:, :, 0], axis=1)
    height     = np.max(seq[:, :, 1], axis=1) - np.min(seq[:, :, 1], axis=1)
    bbox_ratio = (height / (width + 1e-8)).reshape(T, 1)

    # 수직 이동량 (2) — 전체 프레임에 동일값 브로드캐스트
    dy_hip_val = float(((l_hip[-1, 1] + r_hip[-1, 1]) -
                        (l_hip[0,  1] + r_hip[0,  1])) / 2.0)
    dy_sh_val  = float(((l_shoulder[-1, 1] + r_shoulder[-1, 1]) -
                        (l_shoulder[0,  1] + r_shoulder[0,  1])) / 2.0)
    dy_hip = np.full((T, 1), dy_hip_val, dtype=np.float32)
    dy_sh  = np.full((T, 1), dy_sh_val,  dtype=np.float32)

    # 손목 속도 XY (4)
    v_l = np.diff(l_wrist, axis=0, prepend=l_wrist[:1, :])
    v_r = np.diff(r_wrist, axis=0, prepend=r_wrist[:1, :])

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
    ]).astype(np.float32)  # (60, 114)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 포즈 피처 추출기 (raw xyz 추출 + confidence 체크)
# ─────────────────────────────────────────────────────────────────────────────
class PoseFeatureExtractor:
    KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

    def extract_raw_xyz(self, landmarks) -> np.ndarray | None:
        """MediaPipe landmarks → (33, 3) float32"""
        if landmarks is None:
            return None
        lm = landmarks.landmark
        if len(lm) != 33:
            return None
        return np.array(
            [[lm[i].x, lm[i].y, lm[i].z] for i in range(33)],
            dtype=np.float32
        )

    def check_confidence(self, landmarks, threshold: float = 0.5) -> bool:
        if landmarks is None:
            return False
        lm  = landmarks.landmark
        vis = [lm[i].visibility for i in self.KEY_LANDMARKS]
        return float(np.mean(vis)) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. 실시간 추론 엔진
# ─────────────────────────────────────────────────────────────────────────────
class ActionInferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        vote_window: int = 5,
        conf_threshold: float = 0.7,
    ):
        self.device         = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vote_window    = vote_window
        self.conf_threshold = conf_threshold

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = ModelV6_MultiScale_GRU().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # 학습 통계 — shape (1, 1, 114)
        self.X_mean = torch.tensor(ckpt["X_mean"], dtype=torch.float32).to(self.device)
        self.X_std  = torch.tensor(ckpt["X_std"],  dtype=torch.float32).to(self.device)

        # deque에 (33, 3) raw xyz 저장
        self.raw_window:  collections.deque = collections.deque(maxlen=SEQ_LEN)
        self.vote_buffer: collections.deque = collections.deque(maxlen=vote_window)
        self.latencies:   collections.deque = collections.deque(maxlen=100)

        print(f"[ActionInference] 로드 완료: {os.path.basename(model_path)}")
        print(f"[ActionInference] device={self.device} | vote={vote_window} | conf_th={conf_threshold}")

    def push_frame(self, raw_xyz: np.ndarray) -> dict | None:
        """
        Args:
            raw_xyz: (33, 3) float32 — MediaPipe raw xyz

        Returns:
            {"class", "index", "conf", "voted_class", "latency_ms"} or None
        """
        self.raw_window.append(raw_xyz)
        if len(self.raw_window) < SEQ_LEN:
            return None

        t0 = time.perf_counter()

        # 전처리 → (60, 114)
        feat_np = preprocess_window(np.array(self.raw_window, dtype=np.float32))

        # 정규화 → 텐서
        x = torch.from_numpy(feat_np).unsqueeze(0).to(self.device)  # (1, 60, 114)
        x = (x - self.X_mean) / (self.X_std + 1e-8)

        # 추론
        with torch.inference_mode():
            probs = F.softmax(self.model(x), dim=1)[0].cpu().numpy()

        pred_idx   = int(probs.argmax())
        pred_conf  = float(probs[pred_idx])
        pred_class = CLASSES[pred_idx]

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(latency_ms)

        if pred_conf >= self.conf_threshold:
            self.vote_buffer.append(pred_idx)

        return {
            "class":       pred_class,
            "index":       pred_idx,
            "conf":        pred_conf,
            "voted_class": self._voted(),
            "latency_ms":  latency_ms,
        }

    def _voted(self) -> str:
        if not self.vote_buffer:
            return "unknown"
        return CLASSES[int(np.bincount(list(self.vote_buffer), minlength=NUM_CLASSES).argmax())]

    @property
    def avg_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    @property
    def buffer_fill(self) -> int:
        return len(self.raw_window)

    def reset(self):
        self.raw_window.clear()
        self.vote_buffer.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROS2 노드
# ─────────────────────────────────────────────────────────────────────────────
def _make_ros2_node():
    """ROS2가 설치된 환경에서만 임포트."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Int32
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    class ActionRecognitionNode(Node):
        """
        ROS2 Humble 행동인식 노드.

        퍼블리시:
          /action_recognition     (String): JSON {"class", "conf", "latency_ms"}
          /action_recognition_raw (Int32):  클래스 인덱스 (-1 = unknown)
        """
        def __init__(self):
            super().__init__("action_recognition_node")

            self.declare_parameter("model_path",     "")
            self.declare_parameter("camera_id",      0)
            self.declare_parameter("conf_threshold", 0.7)
            self.declare_parameter("vote_window",    5)
            self.declare_parameter("show_window",    False)
            self.declare_parameter("timer_hz",       15)

            model_path     = self.get_parameter("model_path").value
            camera_id      = self.get_parameter("camera_id").value
            conf_threshold = self.get_parameter("conf_threshold").value
            vote_window    = self.get_parameter("vote_window").value
            self.show_win  = self.get_parameter("show_window").value
            hz             = self.get_parameter("timer_hz").value

            self.pub_str = self.create_publisher(String, "/action_recognition",     10)
            self.pub_int = self.create_publisher(Int32,  "/action_recognition_raw", 10)
            self.bridge  = CvBridge()

            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.extractor = PoseFeatureExtractor()
            self.engine    = ActionInferenceEngine(
                model_path, conf_threshold=conf_threshold, vote_window=vote_window
            )

            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.timer      = self.create_timer(1.0 / hz, self._cb)
            self.frame_cnt  = 0
            self.get_logger().info(f"ActionRecognitionNode 시작 | hz={hz}")

        def _cb(self):
            ret, frame = self.cap.read()
            if not ret:
                return
            self.frame_cnt += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            mp_res = self.mp_pose.process(rgb)
            rgb.flags.writeable = True

            raw = None
            if mp_res.pose_landmarks and \
               self.extractor.check_confidence(mp_res.pose_landmarks):
                raw = self.extractor.extract_raw_xyz(mp_res.pose_landmarks)

            result = self.engine.push_frame(raw) if raw is not None else None

            if result:
                # String 퍼블리시
                msg       = String()
                msg.data  = json.dumps({
                    "class":      result["voted_class"],
                    "raw_class":  result["class"],
                    "conf":       round(result["conf"], 4),
                    "latency_ms": round(result["latency_ms"], 2),
                })
                self.pub_str.publish(msg)

                # Int32 퍼블리시
                mi      = Int32()
                mi.data = CLASSES.index(result["voted_class"]) \
                          if result["voted_class"] in CLASSES else -1
                self.pub_int.publish(mi)

                if self.frame_cnt % 100 == 0:
                    self.get_logger().info(
                        f"voted={result['voted_class']} | "
                        f"conf={result['conf']:.3f} | "
                        f"avg_lat={self.engine.avg_latency:.1f}ms"
                    )

            if self.show_win:
                if mp_res.pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, mp_res.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS
                    )
                if result:
                    voted = result["voted_class"]
                    color = (0, 0, 255) if voted == "A053" else (0, 255, 0)
                    cv2.putText(
                        frame, f"{LABEL_MAP.get(voted, voted)} ({result['conf']:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
                    )
                # 버퍼 진행 바
                bw = int(640 * self.engine.buffer_fill / SEQ_LEN)
                cv2.rectangle(frame, (0, 472), (bw, 480), (100, 200, 100), -1)
                cv2.imshow("Action Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rclpy.shutdown()

        def destroy_node(self):
            self.cap.release()
            self.mp_pose.close()
            cv2.destroyAllWindows()
            super().destroy_node()

    return ActionRecognitionNode, rclpy


# ─────────────────────────────────────────────────────────────────────────────
# 6. 독립 실행 테스트
# ─────────────────────────────────────────────────────────────────────────────
def standalone_test(model_path: str, camera_id: int = 0):
    """
    ROS2 없이 실시간 성능 확인.
    터미널에 레이턴시/FPS/예측 결과 출력.
    """
    print("=" * 60)
    print("Standalone Test (ROS2 없이 실행)")
    print("q 키로 종료")
    print("=" * 60)

    engine     = ActionInferenceEngine(model_path)
    extractor  = PoseFeatureExtractor()
    mp_pose    = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_cnt = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_res = mp_pose.process(rgb)
        rgb.flags.writeable = True

        raw = None
        if mp_res.pose_landmarks and extractor.check_confidence(mp_res.pose_landmarks):
            raw = extractor.extract_raw_xyz(mp_res.pose_landmarks)

        result = engine.push_frame(raw) if raw is not None else None

        # 시각화
        if mp_res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, mp_res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        if result:
            voted = result["voted_class"]
            color = (0, 0, 255) if voted == "A053" else (0, 255, 0)
            cv2.putText(
                frame,
                f"{LABEL_MAP.get(voted, voted)} ({result['conf']:.2f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
            )
            print(
                f"\r[{frame_cnt:5d}] voted={voted:4s} | raw={result['class']:4s} | "
                f"conf={result['conf']:.3f} | "
                f"lat={result['latency_ms']:.1f}ms | "
                f"avg={engine.avg_latency:.1f}ms | "
                f"FPS={frame_cnt/(time.time()-t_start):.1f}",
                end="",
            )

        # 버퍼 진행 바
        bw = int(640 * engine.buffer_fill / SEQ_LEN)
        cv2.rectangle(frame, (0, 472), (bw, 480), (100, 200, 100), -1)
        cv2.putText(frame, f"Buffer {engine.buffer_fill}/{SEQ_LEN}",
                    (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    mp_pose.close()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    print(f"\n\n평균 추론 레이턴시: {engine.avg_latency:.2f}ms")
    print(f"평균 FPS: {frame_cnt / elapsed:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 엔트리포인트
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    NodeClass, rclpy = _make_ros2_node()
    rclpy.init(args=args)
    node = NodeClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    standalone_test(
        model_path = r"D:\models\V6__MultiScale_GRU.pth",
        camera_id  = 0
    )