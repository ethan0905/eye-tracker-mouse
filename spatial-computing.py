"""Eye‑controlled mouse + hand‑pinch click & scroll (MediaPipe)
================================================================
• **Eye tracking** – MediaPipe FaceMesh (iris‑refined) → 2‑D quadratic mapping (16‑point grid).
• **Hand tracking** – MediaPipe Hands detects thumb‑index **pinch**:
    – Single pinch → **click**  (unless still calibrating)
    – Pinch‑hold + hand move → **scroll** (gentle)
    – Double‑pinch (≤0.4 s) → toggle control (pause / resume eye‑mouse)
• **NEW**: blue dots show **every FaceMesh landmark (468)** so observers can see what the model uses (green dot = iris centre).
• **NEW**: during calibration you can validate a red target with a **pinch** in addition to pressing **SPACE**.
• **Re‑calibrate gaze**: ⌘ (Command) key (macOS) or **R** fallback.
• **Q** to quit.

Dependencies
------------
```bash
python3 -m pip install opencv-python mediapipe pyautogui numpy
python3 -m pip install pynput  # optional – enables ⌘ shortcut on macOS
```
"""

from __future__ import annotations
import cv2, math, json, pathlib, time
import mediapipe as mp
import numpy as np
import pyautogui

# ─────────────────────────── optional CMD listener ─────────────────────────
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("pynput not installed – use the 'R' key in the window to recalibrate.")
# ───────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Configuration
GRID_SIZE              = 4      # 4 × 4 calibration grid → 16 targets
SMOOTH_ALPHA           = 0.3    # Exponential smoothing for cursor
CALIB_PATH             = pathlib.Path(__file__).with_name('gaze_calib_poly.json')
POINT_RADIUS           = 10     # radius of calibration target (px)
PINCH_THRESHOLD_PX     = 40     # max thumb–index distance for a pinch (px)
COOLDOWN_FRAMES        = 6      # ignore frames after click/pinch
SCROLL_DEADBAND_PX     = 2      # ignore tiny hand jitter while pinched
SCROLL_SENSITIVITY     = 0.4    # scroll speed (lower = gentler)
DOUBLE_PINCH_WINDOW_MS = 400    # ms window for double‑pinch toggle

# Face‑dot rendering
SHOW_FACE_DOTS   = True         # draw landmarks
FACE_DOT_RADIUS  = 1            # radius for each landmark dot
FACE_DOT_COLOR   = (255, 255, 255)  # BGR (blue)
FACE_DOT_INDICES = None         # None → draw all 468, or supply a list for subset

mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# ---------------------------------------------------------------------------
# Polynomial mapping helpers (2‑D quadratic → 6 parameters each axis)

def _design_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(x), x, y, x*y, x**2, y**2))

def fit_poly(src: list[tuple[float, float]], dst: list[tuple[float, float]]):
    src = np.asarray(src); dst = np.asarray(dst)
    Φ   = _design_matrix(src[:,0], src[:,1])
    θx, *_ = np.linalg.lstsq(Φ, dst[:,0], rcond=None)
    θy, *_ = np.linalg.lstsq(Φ, dst[:,1], rcond=None)
    return θx.tolist(), θy.tolist()

def apply_poly(theta, x: float, y: float):
    θx, θy = map(np.asarray, theta)
    Φ      = _design_matrix(np.array([x]), np.array([y]))[0]
    return float(Φ @ θx), float(Φ @ θy)

# ---------------------------------------------------------------------------
class EyeHandMouse:
    def __init__(self):
        # Screen
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.PAUSE    = 0
        pyautogui.FAILSAFE = False

        # Models
        self.face  = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open webcam')

        # Calibration state
        self.theta       = self._load_calibration()
        self.sx_filt     = self.sy_filt = None
        self.targets     = [(self.screen_w*c/(GRID_SIZE-1), self.screen_h*r/(GRID_SIZE-1))
                            for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        self.calib_src: list[tuple[float, float]] = []
        self.calib_dst: list[tuple[float, float]] = []
        self.idx = 0

        # Interaction state
        self.cooldown        = 0
        self.pinch_active    = False
        self.scroll_prev_y   = None
        self.last_pinch_time = 0.0
        self.control_enabled = True
        self.eye_pos         = (0.0, 0.0)

        # Optional CMD listener
        if HAS_PYNPUT:
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.daemon = True
            self.listener.start()
        else:
            self.listener = None

    # ───────────────────── calibration persistence ───────────────────────
    def _load_calibration(self):
        if CALIB_PATH.exists():
            try:
                data = json.loads(CALIB_PATH.read_text())
                if isinstance(data, list) and len(data) == 2:
                    print(f'Loaded calibration from {CALIB_PATH}')
                    return data
            except json.JSONDecodeError:
                pass
        return None

    def _save_calibration(self):
        try:
            CALIB_PATH.write_text(json.dumps(self.theta))
            print(f'✔ Calibration saved to {CALIB_PATH}')
        except IOError as e:
            print(f'⚠ Could not save calibration: {e}')

    # ───────────────────── key‑handlers ──────────────────────────────────
    def _on_key_press(self, key):
        if key == getattr(keyboard.Key, 'cmd', None):
            print('↺ Recalibration triggered by CMD')
            self._start_recalibration()

    def _start_recalibration(self):
        self.theta = None
        self.calib_src.clear(); self.calib_dst.clear(); self.idx = 0

    # ───────────────────── main loop ─────────────────────────────────────
    def run(self):
        print('\n'.join([
            'SPACE **or** Pinch on each red dot to calibrate.',
            'After calibration: Pinch = click / scroll, double‑pinch = pause/resume.',
            'Q = quit.'
        ]))

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print('⚠ Failed to read from webcam. Exiting.')
                break
            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_res  = self.face.process(rgb)
            hands_res = self.hands.process(rgb)

            # ───────── gaze processing ─────────
            if face_res.multi_face_landmarks:
                lm = face_res.multi_face_landmarks[0].landmark

                # Draw facial landmarks
                if SHOW_FACE_DOTS:
                    idx_iter = range(468) if FACE_DOT_INDICES is None else FACE_DOT_INDICES
                    for idx in idx_iter:
                        p = lm[idx]
                        cv2.circle(frame, (int(p.x*w), int(p.y*h)), FACE_DOT_RADIUS, FACE_DOT_COLOR, -1)

                # Iris centre (two refined landmarks)
                ex = (lm[473].x + lm[468].x) / 2
                ey = (lm[473].y + lm[468].y) / 2
                self.eye_pos = (ex, ey)
                cv2.circle(frame, (int(ex*w), int(ey*h)), 3, (0, 255, 0), -1)

                # Move cursor if calibrated & control enabled
                if self.theta is not None and self.control_enabled:
                    sx, sy = apply_poly(self.theta, ex, ey)
                    if self.sx_filt is None:
                        self.sx_filt, self.sy_filt = sx, sy
                    else:
                        self.sx_filt = SMOOTH_ALPHA * sx + (1-SMOOTH_ALPHA) * self.sx_filt
                        self.sy_filt = SMOOTH_ALPHA * sy + (1-SMOOTH_ALPHA) * self.sy_filt
                    pyautogui.moveTo(self.sx_filt, self.sy_filt, _pause=False)

            # ───────── hand / pinch processing ─────────
            if self.cooldown > 0:
                self.cooldown -= 1

            if hands_res.multi_hand_landmarks:
                hand = hands_res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

                # Thumb tip (4) & index tip (8)
                l4 = hand.landmark[4]
                l8 = hand.landmark[8]
                dist = math.hypot((l4.x - l8.x) * w, (l4.y - l8.y) * h)

                if dist < PINCH_THRESHOLD_PX:  # ── PINCH DETECTED
                    if not self.pinch_active:
                        now = time.time()
                        double_pinch = ((now - self.last_pinch_time) * 1000 < DOUBLE_PINCH_WINDOW_MS)
                        self.pinch_active = True
                        self.scroll_prev_y = l8.y * h

                        # CALIBRATION pinch --------------------------------
                        if self.theta is None and self.idx < len(self.targets):
                            self._capture_calibration_sample(w, h)
                            self.cooldown = COOLDOWN_FRAMES
                        else:
                            if double_pinch and self.theta is not None:
                                self.control_enabled = not self.control_enabled
                                print('▶ Control', 'RESUMED' if self.control_enabled else 'PAUSED')
                                self.last_pinch_time = 0
                            else:
                                self.last_pinch_time = now
                                if self.theta is not None and self.control_enabled and self.cooldown == 0:
                                    pyautogui.click()
                                    self.cooldown = COOLDOWN_FRAMES
                                    cv2.circle(frame, (int(l8.x*w), int(l8.y*h)), 20, (0, 0, 255), 2)
                    else:
                        # Pinch held – scroll (after calibration)
                        if self.theta is not None and self.control_enabled:
                            cur_y = l8.y * h
                            dy    = self.scroll_prev_y - cur_y
                            if abs(dy) > SCROLL_DEADBAND_PX:
                                pyautogui.scroll(int(dy * SCROLL_SENSITIVITY))
                                self.scroll_prev_y = cur_y
                else:
                    # Pinch released
                    self.pinch_active = False
                    self.scroll_prev_y = None

            # ───────── draw calibration target ─────────
            if self.theta is None and self.idx < len(self.targets):
                tx, ty = self.targets[self.idx]
                cv2.circle(frame, (int(tx / self.screen_w * w), int(ty / self.screen_h * h)),
                           POINT_RADIUS, (0, 0, 255), 2)

            # ───────── overlay status text ─────────
            if not self.control_enabled and self.theta is not None:
                cv2.putText(frame, 'PAUSED – double‑pinch to resume', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.theta is None:
                cv2.putText(frame, 'CALIBRATION – pinch or SPACE on red dots', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ───────── window / key handling ─────────
            cv2.imshow('Eye & Hand Mouse', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                print('↺ Recalibration started (R key)')
                self._start_recalibration()
                continue
            if key == ord(' ') and self.theta is None and face_res.multi_face_landmarks:
                self._capture_calibration_sample(w, h)

        # ------------------------------------------------------------------
        self._cleanup()

    # ───────────────────── helper methods ────────────────────────────────
    def _capture_calibration_sample(self, w: int, h: int):
        """Capture eye→screen mapping sample and advance to next target."""
        ex, ey = self.eye_pos
        self.calib_src.append((ex, ey))
        self.calib_dst.append(self.targets[self.idx])
        # visual feedback
        tx, ty = self.targets[self.idx]
        cv2.circle(
            img := np.zeros((1, 1, 3), np.uint8),  # dummy img just for colour const
            (0, 0), POINT_RADIUS+4, (0, 255, 255), 2
        )  # noqa – placeholder to keep pylint quiet
        self.idx += 1
        if self.idx == len(self.targets):
            self.theta = fit_poly(self.calib_src, self.calib_dst)
            self._save_calibration()
            print('✔ Calibration complete!')

    def _cleanup(self):
        self.cap.release()
        self.face.close(); self.hands.close()
        if self.listener:
            self.listener.stop()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        EyeHandMouse().run()
    except KeyboardInterrupt:
        print('\nInterrupted by user – exiting.')
