# spatial computing brought to Apple computers

you must have python 3.12 version!
```
(base) ➜  ~ source ~/Desktop/eye-hand-env/bin/activate

((eye-mouse-env) ) (base) ➜  ~ python -V          # --> Python 3.12.x (arm64)
Python 3.12.11

((eye-mouse-env) ) (base) ➜  ~ pip install opencv-python mediapipe==0.10.21 pyautogui

((eye-mouse-env) ) (base) ➜  ~ python - <<'PY'
import sys, cv2, mediapipe as mp, pyautogui
print("Python :", sys.version)
print("OpenCV :", cv2.__version__)
print("MediaPipe :", mp.__version__)
print("PyAutoGUI :", pyautogui.__version__)
PY

Python : 3.12.11 (main, Jun  3 2025, 15:41:47) [Clang 17.0.0 (clang-1700.0.13.3)]
OpenCV : 4.11.0
MediaPipe : 0.10.21
PyAutoGUI : 0.9.54

((eye-mouse-env) ) (base) ➜  ~ python merge-eye-hand.py
```
