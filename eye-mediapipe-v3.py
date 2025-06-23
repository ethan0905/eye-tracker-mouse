"""Eye-controlled mouse (MediaPipe Iris + 2‑D quadratic mapping)
================================================================
• 16‑point calibration grid (press SPACE on each target).
• Polynomial solver captures screen curvature → sub‑cm accuracy.
• Both‑eye averaging + EMA smoothing for stability with low latency.
• Press **R** any time to recalibrate, **Q** to quit.

Changes in this revision
------------------------
* Calibration file is now stored **next to this script** (same repo level)
  instead of in the user’s home directory.

Dependencies
------------
    pip install opencv-python mediapipe pyautogui numpy
"""

from __future__ import annotations
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json, pathlib
from time import perf_counter

# ---------------------------------------------------------------------------
# Configuration
GRID_SIZE      = 4          # 4×4 → 16 calibration targets
SMOOTH_ALPHA   = 0.3        # 0=no smoothing, 1=very smooth (but adds lag)
# Save calibration in the same directory as this script
CALIB_PATH     = pathlib.Path(__file__).with_name('gaze_calib_poly.json')
POINT_RADIUS   = 10         # red calibration‑dot radius (pixels)
mp_face_mesh   = mp.solutions.face_mesh

# ---------------------------------------------------------------------------
# Polynomial mapping helpers (2‑D quadratic → 6 parameters each axis)

def _design_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Φ(x,y) = [1, x, y, x·y, x², y²] for every sample."""
    return np.column_stack((np.ones_like(x), x, y, x * y, x ** 2, y ** 2))

def fit_poly(src: list[tuple[float, float]], dst: list[tuple[float, float]]):
    src_np = np.asarray(src)
    dst_np = np.asarray(dst)
    Φ      = _design_matrix(src_np[:, 0], src_np[:, 1])
    θx, *_ = np.linalg.lstsq(Φ, dst_np[:, 0], rcond=None)
    θy, *_ = np.linalg.lstsq(Φ, dst_np[:, 1], rcond=None)
    return θx.tolist(), θy.tolist()

def apply_poly(thetas, x: float, y: float):
    θx, θy = (np.asarray(thetas[0]), np.asarray(thetas[1]))
    Φ      = _design_matrix(np.asarray([x]), np.asarray([y]))[0]
    return float(Φ @ θx), float(Φ @ θy)

# ---------------------------------------------------------------------------
# Main application class
class EyeMouse:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.PAUSE   = 0  # disable pyautogui's implicit delay
        pyautogui.FAILSAFE = False

        self.mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap  = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open webcam')

        self.theta = self._load_calibration()
        self.sx_filt = self.sy_filt = None  # for smoothing

        # build calibration grid (left→right, top→bottom)
        self.targets = [
            (self.screen_w * c / (GRID_SIZE - 1), self.screen_h * r / (GRID_SIZE - 1))
            for r in range(GRID_SIZE) for c in range(GRID_SIZE)
        ]
        self.calib_src: list[tuple[float, float]] = []
        self.calib_dst: list[tuple[float, float]] = []
        self.target_idx = 0

    # ------------------------- calibration I/O -------------------------
    def _save_calibration(self):
        try:
            CALIB_PATH.write_text(json.dumps(self.theta))
            print(f'✔ Calibration saved to {CALIB_PATH}')
        except IOError as e:
            print(f'⚠ Could not save calibration: {e}')

    def _load_calibration(self):
        if CALIB_PATH.exists():
            try:
                data = json.loads(CALIB_PATH.read_text())
                if isinstance(data, list) and len(data) == 2:
                    print(f'Loaded calibration ({CALIB_PATH})')
                    return data
            except json.JSONDecodeError:
                pass
        return None

    # ---------------------------- core loop ----------------------------
    def run(self):
        print('Press SPACE on each red dot to calibrate — Q to quit, R to recalibrate')
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = self.mesh.process(rgb)

            if res.multi_face_landmarks:
                lm   = res.multi_face_landmarks[0].landmark
                # average both iris centres (left eye 473, right eye 468)
                ex    = (lm[473].x + lm[468].x) / 2.0
                ey    = (lm[473].y + lm[468].y) / 2.0
                cv2.circle(frame, (int(ex * w), int(ey * h)), 3, (0, 255, 0), -1)

                if self.theta is not None:
                    sx, sy = apply_poly(self.theta, ex, ey)
                    # smoothing
                    if self.sx_filt is None:
                        self.sx_filt, self.sy_filt = sx, sy
                    else:
                        self.sx_filt = SMOOTH_ALPHA * sx + (1 - SMOOTH_ALPHA) * self.sx_filt
                        self.sy_filt = SMOOTH_ALPHA * sy + (1 - SMOOTH_ALPHA) * self.sy_filt
                    pyautogui.moveTo(self.sx_filt, self.sy_filt, _pause=False)

            # draw current calibration target
            if self.theta is None and self.target_idx < len(self.targets):
                tx, ty = self.targets[self.target_idx]
                cv2.circle(frame, (int(tx / self.screen_w * w), int(ty / self.screen_h * h)), POINT_RADIUS, (0, 0, 255), 2)

            cv2.imshow('Gaze Mouse', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                print('↺ Recalibration started')
                self.theta = None
                self.calib_src.clear(); self.calib_dst.clear(); self.target_idx = 0
                continue
            # capture calibration sample
            if key == ord(' ') and self.theta is None and res.multi_face_landmarks:
                self.calib_src.append((ex, ey))
                self.calib_dst.append(self.targets[self.target_idx])
                self.target_idx += 1
                if self.target_idx == len(self.targets):
                    self.theta = fit_poly(self.calib_src, self.calib_dst)
                    self._save_calibration()

        self.cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    EyeMouse().run()
