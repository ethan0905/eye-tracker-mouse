"""Eye‑controlled mouse + hand‑pinch click & scroll (MediaPipe)
================================================================
• Eye tracking: MediaPipe FaceMesh iris → 2‑D quadratic mapping (16‑point grid).
• Hand tracking: MediaPipe Hands detects thumb‑index **pinch**:
    – **Single pinch** → mouse **click**  (unless calibrating, see below).
    – **Pinch‑hold + hand move** → page **scroll** (now gentler).
    – **Double‑pinch** (two pinches within 0.4 s) → **toggle control** (pause / resume eye‑mouse);
      use this to hand control back to your normal mouse/trackpad.
• **NEW**: face/eye **tracking dots** rendered in the webcam preview (subset of FaceMesh landmarks + iris centre).
• **NEW**: During calibration, you can validate a dot with a **pinch** in addition to pressing **SPACE**.
• NEW: full hand skeleton ("web") rendered on the webcam preview.
• **Re‑calibrate gaze**: press **⌘ Command** (macOS) — or **R** as a fallback.
• Press **Q** to quit.

Dependencies
------------
    pip install opencv-python mediapipe pyautogui numpy    # required
    pip install pynput                                    # optional (⌘ listener)
"""

from __future__ import annotations
import cv2, math, json, pathlib, time
import mediapipe as mp
import numpy as np
import pyautogui

# ╭───────────────────────────── optional CMD listener ─────────────────────╮
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    print("pynput not installed – use the 'R' key in the window to recalibrate.")
# ╰──────────────────────────────────────────────────────────────────────────╯

# ---------------------------------------------------------------------------
# Configuration
GRID_SIZE              = 4      # 4×4 calibration dots → 16 points
SMOOTH_ALPHA           = 0.3    # Exponential smoothing for cursor
CALIB_PATH             = pathlib.Path(__file__).with_name('gaze_calib_poly.json')
POINT_RADIUS           = 10     # px radius of calibration dot
PINCH_THRESHOLD_PX     = 40     # max thumb–index distance to count as pinch
COOLDOWN_FRAMES        = 6      # frames to ignore after a click/pinch
SCROLL_DEADBAND_PX     = 2      # ignore very tiny hand jitter while pinched
SCROLL_SENSITIVITY     = 0.4    # ← lower = gentler scroll
DOUBLE_PINCH_WINDOW_MS = 400    # two pinches inside this window → toggle control
SHOW_FACE_DOTS         = True   # render tracking dots on face
FACE_DOT_RADIUS        = 1      # px radius of each face dot
FACE_DOT_COLOR         = (255, 0, 0)  # BGR – blue dots
FACE_DOT_INDICES       = [1, 33, 133, 362, 263, 61, 291, 199]  # subset of FaceMesh indices

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
        # --- screen ---
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.PAUSE    = 0
        pyautogui.FAILSAFE = False

        # --- ML models ---
        self.face  = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

        # --- camera ---
        self.cap   = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open webcam')

        # --- calibration ---
        self.theta       = self._load_calibration()
        self.sx_filt     = self.sy_filt = None
        self.targets     = [(self.screen_w*c/(GRID_SIZE-1), self.screen_h*r/(GRID_SIZE-1))
                            for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        self.calib_src: list[tuple[float,float]] = []
        self.calib_dst: list[tuple[float,float]] = []
        self.idx         = 0

        # --- interaction state ---
        self.cooldown        = 0      # pinch click cooldown (frames)
        self.pinch_active    = False
        self.scroll_prev_y   = None
        self.last_pinch_time = 0.0    # for double‑pinch detection
        self.control_enabled = True   # toggled by double‑pinch
        self.eye_pos         = (0.0, 0.0)  # most recent eye centre (ex, ey)

        # --- CMD listener (optional) ---
        if HAS_PYNPUT:
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.daemon = True
            self.listener.start()
        else:
            self.listener = None

    # ------------- calibration persistence ----------------
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

    # ------------- CMD-key handler -------------------------
    def _on_key_press(self, key):
        # macOS only: keyboard.Key.cmd
        if key == getattr(keyboard.Key, 'cmd', None):
            print('↺ Recalibration triggered by CMD')
            self._start_recalibration()

    def _start_recalibration(self):
        self.theta = None
        self.calib_src.clear(); self.calib_dst.clear(); self.idx = 0

    # ------------- main loop ------------------------------
    def run(self):
        print('SPACE or Pinch on each red dot to calibrate.  Pinch = click / scroll after calibration.  Double‑pinch toggles control.')
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_res  = self.face.process(rgb)
            hands_res = self.hands.process(rgb)

            # ---- gaze processing ---------------------------------------
            if face_res.multi_face_landmarks:
                lm = face_res.multi_face_landmarks[0].landmark

                # --- face / eye tracking dots ---------------------------
                if SHOW_FACE_DOTS:
                    for idx in FACE_DOT_INDICES:
                        p = lm[idx]
                        cv2.circle(frame, (int(p.x*w), int(p.y*h)), FACE_DOT_RADIUS, FACE_DOT_COLOR, -1)

                # --- iris centre ---------------------------------------
                ex = (lm[473].x + lm[468].x)/2
                ey = (lm[473].y + lm[468].y)/2
                self.eye_pos = (ex, ey)  # store for pinch‑calibration
                cv2.circle(frame, (int(ex*w), int(ey*h)), 3, (0,255,0), -1)

                # --- control cursor if calibrated ----------------------
                if self.theta is not None and self.control_enabled:
                    sx, sy = apply_poly(self.theta, ex, ey)
                    if self.sx_filt is None:
                        self.sx_filt, self.sy_filt = sx, sy
                    else:
                        self.sx_filt = SMOOTH_ALPHA*sx + (1-SMOOTH_ALPHA)*self.sx_filt
                        self.sy_filt = SMOOTH_ALPHA*sy + (1-SMOOTH_ALPHA)*self.sy_filt
                    pyautogui.moveTo(self.sx_filt, self.sy_filt, _pause=False)

            # ---- hand drawing & pinch‑click / scroll --------------------
            if self.cooldown > 0:
                self.cooldown -= 1

            if hands_res.multi_hand_landmarks:
                hand = hands_res.multi_hand_landmarks[0]
                # draw skeleton
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

                l4 = hand.landmark[4]   # thumb tip
                l8 = hand.landmark[8]   # index tip
                dist = math.hypot((l4.x-l8.x)*w, (l4.y-l8.y)*h)

                if dist < PINCH_THRESHOLD_PX:               # ── PINCH DETECTED
                    if not self.pinch_active:
                        # ── pinch just started
                        now = time.time()
                        # Detect double‑pinch -------------------------------------------------
                        if (now - self.last_pinch_time) * 1000 < DOUBLE_PINCH_WINDOW_MS and self.theta is not None:
                            self.control_enabled = not self.control_enabled
                            state = 'RESUMED' if self.control_enabled else 'PAUSED'
                            print(f'▶ Control {state} by double‑pinch')
                            self.last_pinch_time = 0  # reset to avoid triple triggers
                        else:
                            self.last_pinch_time = now
                        # --------------------------------------------------------------------
                        self.pinch_active  = True
                        self.scroll_prev_y = l8.y * h

                        # ── CALIBRATION PINCH ----------------------------------------------
                        if self.theta is None and self.idx < len(self.targets):
                            # ensure we have a recent eye position
                            ex, ey = self.eye_pos
                            self.calib_src.append((ex, ey))
                            self.calib_dst.append(self.targets[self.idx])
                            # visual feedback: flash yellow circle on captured target
                            tx, ty = self.targets[self.idx]
                            cv2.circle(frame, (int(tx/self.screen_w*w), int(ty/self.screen_h*h)), POINT_RADIUS+4, (0,255,255), 2)
                            self.idx += 1
                            if self.idx == len(self.targets):
                                self.theta = fit_poly(self.calib_src, self.calib_dst)
                                self._save_calibration()
                            self.cooldown = COOLDOWN_FRAMES
                        # --------------------------------------------------------------------
                        elif self.theta is not None and self.control_enabled and self.cooldown == 0:
                            # regular click once calibrated
                            pyautogui.click()
                            self.cooldown = COOLDOWN_FRAMES
                            cv2.circle(frame, (int(l8.x*w), int(l8.y*h)), 20, (0,0,255), 2)
                    else:
                        # ── pinch held: compute scroll delta (only after calibration)
                        if self.theta is not None and self.control_enabled:
                            cur_y = l8.y * h
                            dy    = self.scroll_prev_y - cur_y   # +ve → hand moved up
                            if abs(dy) > SCROLL_DEADBAND_PX:
                                # pyautogui.scroll: +ve = up, −ve = down
                                pyautogui.scroll(int(dy * SCROLL_SENSITIVITY))
                                self.scroll_prev_y = cur_y
                else:
                    # ── pinch released
                    self.pinch_active  = False
                    self.scroll_prev_y = None

            # ---- draw calibration dot ---------------------------------
            if self.theta is None and self.idx < len(self.targets):
                tx, ty = self.targets[self.idx]
                cv2.circle(frame, (int(tx/self.screen_w*w), int(ty/self.screen_h*h)),
                           POINT_RADIUS, (0,0,255), 2)

            # ---- overlay state ----------------------------------------
            if not self.control_enabled and self.theta is not None:
                cv2.putText(frame, 'PAUSED – double‑pinch to resume', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.theta is None:
                cv2.putText(frame, 'CALIBRATION – pinch or SPACE on red dots', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Eye & Hand Mouse', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                print('↺ Recalibration started (R key)')
                self._start_recalibration()
                continue
            # capture calibration sample via SPACE
            if key == ord(' ') and self.theta is None and face_res.multi_face_landmarks:
                self.calib_src.append(self.eye_pos)
                self.calib_dst.append(self.targets[self.idx])
                self.idx += 1
                if self.idx == len(self.targets):
                    self.theta = fit_poly(self.calib_src, self.calib_dst)
                    self._save_calibration()

        # ---- clean‑up ---------------------------------------------------
        self.cap.release()
        self.face.close(); self.hands.close()
        if self.listener:
            self.listener.stop()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    EyeHandMouse().run()
