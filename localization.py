import numpy as np
from tag_unit_test import ATag, readCalibrationFile
import camera
import config

cfg = config.config()

class Localization:     
    # this is the robot to field position
    def get_robot_to_field_position(camera_type="realsense"):
        calib_data = camera.read_calibration_data()

        if camera_type == "realsense":
            cam = camera.RealSenseCamera(640, 360)
            intrinsics = calib_data['intrinsicsRs']
            Cam2Robot = calib_data['rsCamToRobot']
        elif camera_type == "depthai":
            cam = camera.DepthAICamera(640, 360)
            intrinsics = calib_data['intrinsicsDai']
            Cam2Robot = calib_data['daiCamToRobot']
        else:
            raise ValueError("Unsupported camera type")

        camera_params = [
            intrinsics['fx'],
            intrinsics['fy'],
            intrinsics['cx'],
            intrinsics['cy'],
        ]

        atag = ATag(camera_params)

        color_frame, _ = cam.read()
        if color_frame is None:
            print("Failed to capture frame")
            return None

        tag, tag_id, Lm2Cam = atag.getTagAndPose(color_frame, tag_size=cfg.TAG_SIZE_3IN)
        if tag is None:
            print("No valid tag found")
            return None

        Robot2Field = atag.getRobotPoseFromTagPose(Lm2Cam, tag_id, Lm2Cam, Cam2Robot)
        return Robot2Field        
    
    def get_nearest_coke_can_on_enemy_side(): # → (x,y,theta)
        # Returns relative distance (x,y,theta)
        return {
            "x": 5, 
            "y":5,
            "theta": 0
            }
    
    def get_goal_area_landmarks(): # → Struct/Tuple of (x,y,theta)
        # this will be hardcoded instead of figuring this out through localization
        return "landmark placeholder"
    
    def update_map(): # → refresh the map based on the current view of the board
        # not doing this time constraints
        return "map update placeholder"