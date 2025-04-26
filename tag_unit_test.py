import cv2
import numpy as np
import json
import camera
import landmarks_2025 as landmarks

from pyapriltags import Detector
import config 
cfg = config.Config()

def readCalibrationFile(path=cfg.CALIB_PATH):
  calib = np.loadtxt(path, delimiter=",")
  rsCamToRobot = calib[:4,:]
  daiCamToRobot = calib[4:,:]
  return rsCamToRobot, daiCamToRobot

class ATag:
  def __init__(self, camera_params) -> None:
    self.at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1.2,
                          debug=False)# cfg.GEN_DEBUG_IMAGES)
    self.camera_params = camera_params
    pass

  def getTagAndPose(self,color, tag_size=cfg.TAG_SIZE_3IN):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = cfg.TAG_DECISION_MARGIN_THRESHOLD
      min_pose_err = cfg.TAG_POSE_ERROR_THRESHOLD
      max_confidence_tag = None
      for tag in self.tags:
        # print all tag information like pose err and decision margin
        # if tag.tag_id == 6:
          # print(f'tag.decision_margin = {tag.decision_margin}, tag.pose_err = {tag.pose_err}, tag.tag_id = {tag.tag_id}')
          # print(tag.pose_t)
        
        if cfg.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > cfg.TAG_POSE_ERROR_THRESHOLD:
            # print(f'tag.decision_margin = {tag.decision_margin} < {cfg.TAG_DECISION_MARGIN_THRESHOLD}')
            continue
        if tag.decision_margin > max_confidence and tag.pose_err < min_pose_err:
          max_confidence = tag.decision_margin
          min_pose_err = tag.pose_err
          max_confidence_tag = tag
      if max_confidence_tag is not None:
        # tag_pose = max_confidence_tag.pose_t
        tag_id = max_confidence_tag.tag_id
        R = max_confidence_tag.pose_R
        t = max_confidence_tag.pose_t
        # make 4 * 4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return max_confidence_tag, tag_id, T
    return None, None, None
  
  def getRobotPoseFromTagPose(self, tag_pose, tag_id, Lm2Cam, Cam2Robot):
    if tag_pose is None or tag_id is None or Lm2Cam is None:
      # print(f'getRobotPoseFromTagPose: tag_pose = \n{tag_pose}, tag_id = {tag_id}, Lm2Cam = \n{Lm2Cam}')
      return None
    if tag_id in landmarks.map_apriltag_poses:
      # print(f'tag_id = {tag_id} in landmarks.map_apriltag_poses')
      L2F = landmarks.map_apriltag_poses[tag_id]
      Field2Robot = L2F @ np.linalg.inv(Lm2Cam) @ np.linalg.inv(Cam2Robot)
      # print(f'landmarks.map_apriltag_poses[{tag_id}] = \n{landmarks.map_apriltag_poses[tag_id]}')
      # print("Lm2Cam = \n", Lm2Cam)
      # print("Cam2Robot = \n", Cam2Robot)
    #   print("Field2Robot = \n", Field2Robot[:3,3])
    # np.savetxt("debug_transformation.txt", np.vstack((Lm2Cam, Cam2Robot)), delimiter=",")
      return Field2Robot
    else:
      return None

ROBOT_STATE = {"OBJECT_SCAN", "OBJECT_DETECTED", "REACHED_OBJECT_GRAB_LOC", "GRABBED_OBJ", "TARGET_TAG_FOUND", "MOVE_TO_TARGET", "DROP_BALL", "SAFE_LOCATION_TO_SCAN", "STUCK"}




def main():
    # Camera selection
    use_realsense = True  # Set to False to disable RealSense
    use_dai = False       # Set to False to disable DepthAI
    
    # Initialize cameras based on selection
    camRS = None
    camDAI = None
    
    if use_realsense:
        camRS = camera.RealSenseCamera(640, 360)
    if use_dai:
        camDAI = camera.DepthAICamera(640, 360)
    
    if not use_realsense and not use_dai:
        print("Error: No cameras enabled. Please enable at least one camera.")
        return
    
    calib_data = camera.read_calibration_data()
    
    # Initialize camera parameters and transformation matrices
    camera_params = None
    rsCamToRobot = None
    daiCamToRobot = None
    
    if use_realsense:
        intrinsicsRs = calib_data['intrinsicsRs']
        rsCamToRobot = calib_data['rsCamToRobot']
        camera_params = [ 
            intrinsicsRs['fx'],
            intrinsicsRs['fy'],
            intrinsicsRs['cx'],
            intrinsicsRs['cy'],
        ]
    elif use_dai:
        intrinsicsDai = calib_data['intrinsicsDai']
        daiCamToRobot = calib_data['daiCamToRobot']
        camera_params = [
            intrinsicsDai['fx'],
            intrinsicsDai['fy'],
            intrinsicsDai['cx'],
            intrinsicsDai['cy'],
        ]
    
    atag = ATag(camera_params)

    # detect the tag and get the pose
    if cfg.FIELD == "HOME" or cfg.FIELD == "GAME":
        tag_size = cfg.TAG_SIZE_3IN # inches

    def unit_test_atag(camera_params, cam, tag_size, CamToRobot, camera_name):
        if cam is None:
            print(f"{camera_name} camera is disabled")
            return
            
        np.set_printoptions(precision=2, suppress=True)
        atag = ATag(camera_params)
        
        # Define colors for different tag types
        FIELD_COLOR = (0, 255, 0)  # Green for field tags
        OBJ_COLOR = (255, 0, 0)    # Blue for object tags
        CLAW_COLOR = (0, 0, 255)   # Red for claw tags
        
        # Create a named window
        window_name = f"Tag Detection - {camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # Read camera frames
            color_frame, depth_frame = cam.read()
            if color_frame is None:
                print(f"Error: Could not read frame from {camera_name}")
                continue
                
            # Create a copy of the frame for visualization
            vis_frame = color_frame.copy()
            
            # Get all tags
            tags = atag.at_detector.detect(
                cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size)
            
            best_tag = None
            max_confidence = -1

            if tags is not None:
                # First loop: Find the best tag and draw all valid tags
                for tag in tags:
                    # Skip tags that don't meet confidence thresholds
                    if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > cfg.TAG_POSE_ERROR_THRESHOLD:
                        continue
                    
                    # Update best tag if current tag has higher confidence AND is a field tag
                    if tag.tag_id in landmarks.map_apriltag_poses and tag.decision_margin > max_confidence:
                        max_confidence = tag.decision_margin
                        best_tag = tag
                    
                    # Draw the tag corners
                    for idx in range(len(tag.corners)):
                        cv2.line(vis_frame, tuple(tag.corners[idx-1].astype(int)), 
                                tuple(tag.corners[idx].astype(int)), (0, 255, 0), 2)
                    
                    # Determine tag type and color
                    if tag.tag_id in landmarks.map_apriltag_poses:
                        prefix = "field"
                        color = FIELD_COLOR
                    elif tag.tag_id == 29:
                        prefix = "obj"
                        color = OBJ_COLOR
                    elif tag.tag_id == 28:
                        prefix = "claw"
                        color = CLAW_COLOR
                    else:
                        prefix = "unknown"
                        color = (255, 255, 255)  # White for unknown tags
                    
                    # Draw tag ID with prefix and confidence
                    tag_text = f"{prefix}_{tag.tag_id} (conf:{tag.decision_margin:.1f})"
                    cv2.putText(vis_frame, tag_text, 
                               tuple(tag.center.astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # After checking all tags, process the best one found
            if best_tag is not None:
                # Calculate and print robot pose for the best tag
                Lm2Cam = np.eye(4)
                Lm2Cam[0:3, 0:3] = best_tag.pose_R
                Lm2Cam[0:3, 3] = best_tag.pose_t.ravel()
                Robot2Field = atag.getRobotPoseFromTagPose(best_tag, best_tag.tag_id, Lm2Cam, CamToRobot)
                
                if Robot2Field is not None:
                    x = Robot2Field[0,3]
                    y = -Robot2Field[1,3]
                    orientation_rad = np.arccos(Robot2Field[0, 0])
                    orientation_deg = np.degrees(orientation_rad)
                    print(f"Robot2Field orientation_deg: {orientation_deg:.2f} degrees, x: {x:.2f}, y: {y:.2f} = \n{Robot2Field}")
                    # print(f"orientation_deg: {orientation_deg:.2f} degrees, x: {x:.2f}, y: {y:.2f}")
                    # print(f"{camera_name}: {tag.tag_id}, x: {x:.2f}, y: {y:.2f}, confidence: {tag.decision_margin:.2f}, pose_err: {tag.pose_err:.2f}")
                    # print(f"{camera_name}: Best Tag ID: {best_tag.tag_id}, x: {x:.2f}, y: {y:.2f}") #, confidence: {best_tag.decision_margin:.2f}, pose_err: {best_tag.pose_err:.2f}")
                    # print(f"{camera_name}: Best Tag ID: {best_tag.tag_id}, x: {x:.2f}, y: {y:.2f}, orientation: {orientation_deg:.2f} degrees")
            
            # Display the visualization
            cv2.imshow(window_name, vis_frame)
            
            # Break loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyWindow(window_name)

    # Run unit tests for enabled cameras
    if use_realsense:
        unit_test_atag(camera_params, camRS, tag_size, rsCamToRobot, "RealSense")
    if use_dai:
        unit_test_atag(camera_params, camDAI, tag_size, daiCamToRobot, "DepthAI")

if __name__ == "__main__":
    main()