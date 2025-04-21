import cortano
import cv2
import numpy as np
import pyapriltags
from scipy.spatial.transform import Rotation
 
# Parameters for D415 color camera, 640x360 res
cx = 319.128479003906
cy = 176.236511230469
fx = 457.265380859375
fy = 456.629211425781
 
detector = pyapriltags.Detector(families='tag16h5')
def localize(color):
    if color is None: return None

    detected=detector.detect(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=1.5
    )
    cam2claw, cam2coke = None, None
    for tag in detected:
        if tag.decision_margin < 50 or abs(tag.pose_err) > 0.4: continue
        print(f"found tag id = ", tag.tag_id)
        cam_tag_T = np.eye(4)
        cam_tag_T[:3, :3] = tag.pose_R
        cam_tag_T[:3, 3:] = tag.pose_t
        if tag.tag_id == 29:
            print("Set RT for cam2coke")
            # Coke
            cam2coke = cam_tag_T
        elif tag.tag_id == 28:
            print("Set RT for cam2claw")
            # Claw
            cam2claw = cam_tag_T
 
    return cam2coke, cam2claw
 
 
 
    if name == "main":
        camera = cortano.RealsenseCamera()
        print ("init'd realsense")
 
 
    from time import sleep
    c2coke, c2claw = None, None
    for i in range(60):
        color, depth = camera.read()
        cv2.imwrite('color.png', color)
        cam2coke, cam2claw = localize(color)
        if cam2coke is not None:
            c2coke = cam2coke
        if cam2claw is not None:
            c2claw = cam2claw
    if c2coke is None:
        raise Exception("c2coke is None")
    elif c2claw is None:
        raise Exception("c2claw is None")
    claw2coke = np.linalg.inv(c2claw) @ c2coke
    print("Result:", claw2coke)