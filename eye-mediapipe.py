import cv2, mediapipe as mp, pyautogui, numpy as np, json, pathlib, time

# ---------- helpers ----------
def solve_homography(src, dst):
    A=[]
    for (x,y),(X,Y) in zip(src, dst):
        A.extend([[x,y,1,0,0,0,-X*x,-X*y,-X],
                  [0,0,0,x,y,1,-Y*x,-Y*y,-Y]])
    _,_,Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3,3)
    return H / H[2,2]

def apply_H(H, x, y):
    v = H @ np.array([x,y,1])
    return v[0]/v[2], v[1]/v[2]

# ---------- setup ----------
mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
scr_w, scr_h = pyautogui.size()
cap = cv2.VideoCapture(0)

CAL_FILE = pathlib.Path.home()/'.gaze_calib.json'
if CAL_FILE.exists():
    H = np.array(json.loads(CAL_FILE.read_text()))
    calibrated = True
else:
    targets    = [(0,0),(scr_w,0),(scr_w,scr_h),(0,scr_h),(scr_w/2,scr_h/2)]
    calib_pts  = []
    gaze_pts   = []
    idx        = 0
    calibrated = False
    print("Look at target, press SPACE → next target")

while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]              # <-- always defined now
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = mesh.process(rgb)

    if res.multi_face_landmarks:
        lm  = res.multi_face_landmarks[0].landmark
        eye = lm[468]                   # iris centre
        ex, ey = eye.x, eye.y           # 0–1
        cv2.circle(frame,(int(ex*w),int(ey*h)),3,(0,255,0),-1)

        if calibrated:
            sx, sy = apply_H(H, ex, ey)
            pyautogui.moveTo(sx, sy, _pause=False)

    # draw calibration target
    if not calibrated and idx < len(targets):
        tx, ty = targets[idx]
        cv2.circle(frame,(int(tx/scr_w*w),int(ty/scr_h*h)),8,(0,0,255),2)

    cv2.imshow('Gaze Mouse', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' ') and not calibrated and res.multi_face_landmarks:
        calib_pts.append((ex, ey))
        gaze_pts.append(targets[idx])
        idx += 1
        if idx == len(targets):
            H = solve_homography(calib_pts, gaze_pts)
            CAL_FILE.write_text(json.dumps(H.tolist()))
            calibrated = True
            print("✔ Calibration saved – enjoy!")

cap.release(); cv2.destroyAllWindows()

