import camera
import time

####################################################################
import numpy as np
import pyrealsense2 as rs
from typing import Tuple
import cv2
from datetime import datetime
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import logging
#!/usr/bin/env python3
import config

#For tiny yolo
from pathlib import Path
import sys
import depthai as dai

import queue
# create queue of size 10 
daiQueue = queue.Queue(100)


class RealSenseCamera:
  def __init__(self, width=640, height=360, fps=30, debug_mode=True):
    self.debug_mode = debug_mode
    self.initCamera(width, height, fps)

  def initCamera(self, width, height, fps):
    self.width = width
    self.height = height
    self.shape = (height, width)
    self.pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
    profile = self.pipeline.start(config)
    # intel realsense on jetson nano sometimes get misdetected as 2.1 even though it has 3.2 USB
    # profile = self.pipeline.start()
    depth_sensor = profile.get_device().first_depth_sensor()
    self.depth_scale = depth_sensor.get_depth_scale()
    self.align = rs.align(rs.stream.color)
    # Get the intrinsics of the color stream
    self.color_sensor = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
    self.color_intrinsics = self.color_sensor.get_intrinsics()
    self.colorTodepthExtrinsics = self.color_sensor.get_extrinsics_to(profile.get_stream(rs.stream.depth))  
    self.fx = self.color_intrinsics.fx
    self.fy = self.color_intrinsics.fy
    self.cx = self.color_intrinsics.ppx
    self.cy = self.color_intrinsics.ppy
    self.hfov = np.degrees(np.arctan2(self.width  / 2, self.fx)) * 2
    self.vfov = np.degrees(np.arctan2(self.height / 2, self.fy)) * 2
    self.hfov_rad = np.radians(self.hfov)
    self.vfov_rad = np.radians(self.vfov)
    self.hfov_rad_half = self.hfov_rad / 2
    self.vfov_rad_half = self.vfov_rad / 2
    #TODO Get camera to IMU extrinsics
    #depth filtering
    self.min_depth = 0.1
    self.max_depth = 3.0 

  def getColorTodepthExtrinsics(self):
    # Get the extrinsics from color to depth
    extrinsics = self.color_sensor.get_extrinsics_to(self.pipeline.get_active_profile().get_stream(rs.stream.depth))
    # Construct the transformation matrix (3x4)
    R = np.array(extrinsics.rotation).reshape(3, 3)
    T = np.array(extrinsics.translation).reshape(3, 1)

    # Use only the rotation part for the homography matrix
    # and assume the translation in z-axis does not significantly affect the transformation
    homography_matrix = np.eye(3, dtype=np.float64)
    homography_matrix[:2, :2] = R[:2, :2]  # Using only the top-left 2x2 part of the rotation matrix
    homography_matrix[:2, 2] = T[:2, 0]    # Using the x and y components of the translation vector

    return homography_matrix

  def resetCamera(self, width, height):
        # Don't do hw reset as it slows down the camera initialization
    try:
      # Create a context object
      ctx = rs.context()
      # Get a device
      dev = ctx.devices[0]
      # Reset the camera
      dev.hardware_reset()
      self.initCamera(width, height)
    except:
      print("No realsense camera found, or failed to reset.")

  def getCameraIntrinsics(self):
    return (self.fx, self.fy, self.cx, self.cy, self.width, self.height, self.depth_scale)
  
  def capture(self) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Grab frames from the realsense camera
    Args:
        unit (str, optional): inches|meters|raw. Defaults to 'inches'.
    Returns:
        Tuple[bool, np.ndarray, np.ndarray]: status, color image, depth image
    """
    retries = 5
    while retries > 0:
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise ValueError("Empty frames received")
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            return True, color_image, depth_image
        except Exception as e:
            retries -= 1
            print(f"Error capturing frames: {e}, retries left: {retries}")
            time.sleep(0.1)  # brief pause before retrying
    return False, None, None

  def __del__(self):
    try:
      if self.pipeline:
        self.pipeline.stop()
        self.pipeline = None
        # delete cv2 window if it exists
        cv2.destroyAllWindows()
    except:
      pass

  def read(self, scale=False): # will also store in buffer to be read
    ret, color, depth = self.capture()
    if not ret:
      return None, None #np.zeros((self.height, self.width, 3), dtype=np.uint8), \
            #  np.zeros((self.height, self.width), dtype=np.float32)
    if scale:
      depth = depth.astype(np.float32) * self.depth_scale
    # color = np.ascontiguousarray(np.flip(color, axis=-1))

    return color, depth

  def depth2rgb(self, depth):
    if depth.dtype == np.uint16:
      return cv2.applyColorMap(np.sqrt(depth).astype(np.uint8), cv2.COLORMAP_JET)
    else:
      return cv2.applyColorMap(np.floor(np.sqrt(depth / self.depth_scale)).astype(np.uint8), cv2.COLORMAP_JET)

  def view(self):
    color, depth = self.read()
    combined = np.hstack((color, self.depth2rgb(depth)))
    cv2.imshow("combined", combined)
    if cv2.waitKey(1) == 27:
      exit(0)
  
  def viewFilteredDepth(self):
    color, depth = self.read()
    depth = depth.astype(np.float32) * self.depth_scale
    mask = np.bitwise_and(depth > self.min_depth, depth < self.max_depth)
    filtered_depth = np.where(mask, depth, 0)
    filtered_color = np.where(np.tile(mask.reshape(self.height, self.width, 1), (1, 1, 3)), color, 0)
    combined = np.hstack((filtered_color, self.depth2rgb(depth), self.depth2rgb(filtered_depth)))
    cv2.imshow("filtered_color, depth, and filtered depth", combined)
    if cv2.waitKey(1) == 27:
      exit(0)

  def viewDepth(self):
    color, depth = self.read()
    cv2.imshow("depth", self.depth2rgb(depth))
    if cv2.waitKey(1) == 27:
      exit(0)

  def ViewColor(self):
    color, depth = self.read()
    cv2.imshow("color", color)
    if cv2.waitKey(1) == 27:
      exit(0)

  def getFilteredColorBasedOnDepth(self):
    color, depth = self.read()
    depth = depth.astype(np.float32) * self.depth_scale
    mask = np.bitwise_and(depth > self.min_depth, depth < self.max_depth)
    self.filtered_color = np.where(np.tile(mask.reshape(self.height, self.width, 1), (1, 1, 3)), color, 0)
    # cv2.imshow("filtered color", filtered_color)
    # if cv2.waitKey(1) == 27:
    #   exit(0)

  def getTagPose(self, tag_size, specified_tag=255, input_type="color"):
    at_detector = Detector(families='tag16h5',
                           nthreads=4,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=1.2,
                           debug=0)
    camera_params = self.getCameraIntrinsics()
    # Convert tag size from inches to meters for the detector
    tag_size_meters = tag_size * 0.0254  # 1 inch = 0.0254 meters
    
    # detect the tag and get the pose
    while True:
      if input_type == "color":
        color, depth = self.read()
        self.filtered_color = color    # TODO vairable cleanup 
      elif input_type == "filtered_color":
        self.getFilteredColorBasedOnDepth()
      # check if the color image is valid
      if self.filtered_color is None:
        continue
      
      # Convert depth to meters
      depth_meters = depth.astype(np.float32) * self.depth_scale
      
      tags = at_detector.detect(
        cv2.cvtColor(self.filtered_color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size_meters)
      
      if specified_tag != 255:
        for tag in tags:
          if tag.tag_id == specified_tag:
            # Get depth at tag corners
            corners = tag.corners.astype(int)
            corner_depths = []
            for corner in corners:
              x, y = corner
              if 0 <= x < self.width and 0 <= y < self.height:
                corner_depths.append(depth_meters[y, x])
            
            if len(corner_depths) == 4:  # All corners are valid
              # Use average depth of corners
              avg_depth = np.mean(corner_depths)
              
              # Scale translation vector by actual depth
              t = tag.pose_t
              scale_factor = avg_depth / t[2]  # t[2] is the estimated z distance
              t = t * scale_factor
              
              # Convert translation back to inches for consistency
              t = t / 0.0254  # Convert meters to inches
              
              R = tag.pose_R
              # make 4 * 4 transformation matrix
              T = np.eye(4)
              T[0:3, 0:3] = R
              T[0:3, 3] = t.ravel()
              return T
      else:
        for tag in tags:
          if tag.decision_margin > 150: 
            # Convert translation back to inches for all tags
            for tag in tags:
              tag.pose_t = tag.pose_t / 0.0254  # Convert meters to inches
            return tags
          break
        return None

  def hackYdist(self, tag_size, input_type="color"): # other input type is filtered_color
      tags = self.getTagPose(tag_size=tag_size)
      if tags is not None:
        # for tag in tags:
        #   cv2.imshow("color", self.filtered_color)
        #   if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     exit(0)
        #   continue
        for tag in tags:
            if tag.decision_margin < 150:
                continue
            # print the tag pose, tag id, etc well formatted
            # print("Tag Family: ", tag.tag_family)
            print("Tag ID: ", tag.tag_id)
            # print("Tag Hamming Distance: ", tag.hamming)
            # print("Tag Decision Margin: ", tag.decision_margin)
            # print("Tag Homography: ", tag.homography)
            # print("Tag Center: ", tag.center)
            # print("Tag Corners: ", tag.corners)
            # print("Tag Pose: ", tag.pose_R, tag.pose_t)
            # print("Tag Pose Error: ", tag.pose_err)
            # print("Tag Size: ", tag.tag_size)
            print("Tag y distance: ", tag.pose_t[2])
            return tag.pose_t[2] # return y distance Hack
        
  def visTagPose(self, tag_size, input_type="color"): # other input type is filtered_color
      tags = self.getTagPose(tag_size=tag_size)
      if tags is not None:
        # for tag in tags:
        #   cv2.imshow("color", self.filtered_color)
        #   if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     exit(0)
        #   continue
        for tag in tags:
            if tag.decision_margin < 150:
                continue
            # print the tag pose, tag id, etc well formatted
            # print("Tag Family: ", tag.tag_family)
            print("Tag ID: ", tag.tag_id)
            # print("Tag Hamming Distance: ", tag.hamming)
            # print("Tag Decision Margin: ", tag.decision_margin)
            # print("Tag Homography: ", tag.homography)
            # print("Tag Center: ", tag.center)
            # print("Tag Corners: ", tag.corners)
            # print("Tag Pose: ", tag.pose_R, tag.pose_t)
            # print("Tag Pose Error: ", tag.pose_err)
            # print("Tag Size: ", tag.tag_size)
            print("Tag y distance: ", tag.pose_t[2])
            font_scale = 0.5  # Adjust this value for a smaller font size
            font_color = (0, 255, 0)
            font_thickness = 2
            text_offset_y = 30  # Vertical offset between text lines
            # Display tag ID
            cv2.putText(self.filtered_color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Display only the 1 most significant digit for pose_R and pose_t
            pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
            pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
            cv2.putText(self.filtered_color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(self.filtered_color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.imshow("color", self.filtered_color)
        if cv2.waitKey(1) == 27:
          cv2.destroyAllWindows()
          exit(0)

  def getColor(self):
    color, depth = self.read()
    return color
 
def TransformationInverse(T):
   R = T[:3,:3]
   Ri = R.T
   t = T[:3,3:]
   ti = Ri @ -t
   Ti = np.eye(4)
   Ti[:3,:3] = Ri
   Ti[:3,3:] = ti
   return Ti
    
class CalibrateCamera:
 def __init__(self, cam):
    self.camera_params = [cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height]
    self.cam = cam
    return

 def getCamera2Robot(self,tag_size=5, tag_id=0, viz = False):
      # I hand measured the center of the april tag to the robot's claw, i.robot to calibration tag = 16.625 inches
      # tag size = 5 inches
      # T from calibration tag to robot is L2R
      L2R = np.eye(4) # L2R is the transformation from calibration tag to robot
      L2R[0:3, 0] = np.array([1, 0, 0])
      L2R[0:3, 1] = np.array([0, -1, 0])
      L2R[0:3, 2] = np.array([0, 0, -1])
      L2R[0:3, 3] = np.array([0, 16.625, 0])

      print(" landmark to robot L2R = \n", L2R)
      self.L2C= self.landmark2camera(tag_size=tag_size, tag_id=tag_id, viz=viz)
      print(" landmark to camera L2C = \n", self.L2C)
      return  (L2R @ np.linalg.inv(self.L2C)) 
 
 def landmark2camera(self,tag_size=5, tag_id=0, viz=False):
    # make 4 * 4 transformation matrix
    cfg = config.Config()
    T = np.eye(4)
    at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1,
                          debug=0)
    # detect the tag and get the pose
    count = 0
    print("self.camera_params[0:4] = ", self.camera_params[0:4])
    while True:
      count += 1
      self.color, _ = self.cam.read()
      tags = at_detector.detect(
        cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size=tag_size)
      found_tag = False
      for tag in tags:
          if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > cfg.TAG_POSE_ERROR_THRESHOLD: 
            continue
          found_tag = True
          # if tag.tag_id != 0:
          #   continue
          if tag.tag_id != tag_id:
            continue
          # print("Tag ID: ", tag.tag_id)
          # print("Tag Pose: ", tag.pose_R)
          print("Tag translation: ", tag.pose_t)
          R = tag.pose_R
          t = tag.pose_t
          T[0:3, 0:3] = R
          T[0:3, 3] = t.ravel()
          # print("Tag Pose Error: ", tag.pose_err)
          # print("Tag Size: ", tag.tag_size)
      if not viz:
        if count > 10: # this is a hack to get the camera to focus on the tag for dai
          return T
        else:
          continue
      else:
        # np.set_printoptions(precision=2, suppress=True)
        # print(T)
        if not found_tag:
          cv2.imshow("color", self.color)
          if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return T
            # exit(0)
          continue
        for tag in tags:
            if tag.tag_id != tag_id:
                continue
            if tag.decision_margin < 50:
                continue
            font_scale = 0.5  # Adjust this value for a smaller font size
            font_color = (0, 255, 0)
            font_thickness = 2
            text_offset_y = 30  # Vertical offset between text lines
            # Display tag ID
            cv2.putText(self.color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Display only the 1 most significant digit for pose_R and pose_t
            pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
            pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
            cv2.putText(self.color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(self.color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.imshow("color", self.color)
        if cv2.waitKey(1) == 27:
          cv2.destroyAllWindows()
          return T
          # exit(0)


# from vex_serial import VexControl
class DepthAICamera:
  def __init__(self, width=1920, height=1080, viz=False, debug_mode=False, fps=30):
    self.debug_mode = debug_mode
    self.initCamera(width, height, fps)
    self.viz = viz

  # descructor to stop the pipeline
  def __del__(self):
    try:
      cv2.destroyAllWindows()
    except:
      pass

  def initCamera(self, width, height, fps):
    self.width = width
    self.height = height
    # Create pipeline
    self.pipeline = dai.Pipeline()
    # Define source and output
    print("initilying depthai color and depth")
    # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
    extended_disparity = False
    # Better accuracy for longer distance, fractional disparity 32-levels:
    subpixel = True
    # Better handling for occlusions:
    lr_check = True
    self.camRgb = self.pipeline.create(dai.node.ColorCamera)
    self.camRgb.setPreviewSize(width, height)
    self.camRgb.setFps(fps)
    self.camRgb.setInterleaved(True)
    self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    self.StereoDepth = self.pipeline.create(dai.node.StereoDepth)
    self.spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

    self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
    self.xoutRgb.setStreamName("rgb_default")
    self.camRgb.preview.link(self.xoutRgb.input)
    
    self.camLeft = self.pipeline.create(dai.node.MonoCamera)
    self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    self.camLeft.setCamera("left")
    self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
    self.xoutLeft.setStreamName("left")
    self.camLeft.out.link(self.xoutLeft.input)

    self.camRight = self.pipeline.create(dai.node.MonoCamera)
    self.camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    self.camRight.setCamera("right")
    self.xoutRight = self.pipeline.create(dai.node.XLinkOut)
    self.xoutRight.setStreamName("right")
    self.camRight.out.link(self.xoutRight.input)

    # Link left and right cameras to StereoDepth node
    self.camLeft.out.link(self.StereoDepth.left)
    self.camRight.out.link(self.StereoDepth.right)

    self.xoutDisparity = self.pipeline.create(dai.node.XLinkOut)
    self.xoutDisparity.setStreamName("disparity")
    self.xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
    self.xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

    self.xoutSpatialData.setStreamName("spatialData")
    self.xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
    self.StereoDepth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    self.StereoDepth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    self.StereoDepth.setLeftRightCheck(lr_check)
    self.StereoDepth.setExtendedDisparity(extended_disparity)
    self.StereoDepth.setSubpixel(subpixel)
    self.StereoDepth.disparity.link(self.xoutDisparity.input)

    # Connect to device and start pipeline
    self.device = dai.Device(self.pipeline)
    self.qRgb = self.device.getOutputQueue(name="rgb_default", maxSize=2, blocking=False)
    self.qLeft = self.device.getOutputQueue(name="left")
    self.qRight = self.device.getOutputQueue(name="right")
    self.qDepth = self.device.getOutputQueue(name="disparity", maxSize=2, blocking=False)

    self.camera_params = self.getCameraIntrinsics() #width, height)

    self.depth_scale = self.fx

    #warm up the camera to get rid of the first few frames that have low exposure
    for i in range(10):
      if self.qRgb.has():
        self.qRgb.get()
      if self.qDepth.has():
        self.qDepth.get()

  def getCameraIntrinsics(self):
    calibData = self.device.readCalibration()
    import math 
    M_rgb, wid, hgt = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
    self.cx = M_rgb[0][2] * (self.width / wid)
    self.cy = M_rgb[1][2] * (self.height / hgt)
    self.fx = M_rgb[0][0] * (self.width / wid)
    self.fy = M_rgb[1][1] * (self.height / hgt)
    self.focal_length_in_pixels = self.fx
    self.hfov =  2 * 180 / (math.pi) * math.atan(self.width * 0.5 / self.fx)
    self.vfov =  2 * 180 / (math.pi) * math.atan(self.height * 0.5 / self.fy)
    self.baseline = 7.5 #cm
    if True: #self.debug_mode:
      print("OAKD-1 RGB Camera Default intrinsics...")
      print(M_rgb)
      print(wid)
      print(hgt)
      print("cx = ", self.cx)
      print("cy = ", self.cy)
      print("fx = ", self.fx)
      print("fy = ", self.fy)
    return [self.fx, self.fy, self.cx, self.cy]

  def read(self, scale=False): # will also store in buffer to be read
    color = self.qRgb.get().getCvFrame()
    if self.qDepth.has():
      depth = self.qDepth.get().getFrame()
    else:
      depth = np.zeros((self.height, self.width), dtype=np.float32)
    return color, depth 
  
  def view(self):
    while True:
      color, _ = self.read()
      cv2.imshow("color", color)
      if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        exit(0)
      continue

  def getTagPoses(self,tag_size=5, viz=True):
    # make 4 * 4 transformation matrix
    T = np.eye(4)
    at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1,
                          debug=0)
    # detect the tag and get the pose
    count = 0
    print("self.camera_params[0:4] = ", self.camera_params[0:4])
    while True:
      count += 1
      color = self.read()
      tags = at_detector.detect(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size=tag_size)
      found_tag = False
      for tag in tags:
          if tag.decision_margin < 50: 
            continue
          found_tag = True
          R = tag.pose_R
          t = tag.pose_t
          T[0:3, 0:3] = R
          T[0:3, 3] = t.ravel()
      if not viz:
        if count > 10:
          return T
        else:
          continue
      else:
        if not found_tag:
          cv2.imshow("color", color)
          if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return T
          continue
        for tag in tags:
            if tag.decision_margin < 50:
                continue
            font_scale = 0.5  # Adjust this value for a smaller font size
            font_color = (0, 255, 0)
            font_thickness = 2
            text_offset_y = 30  # Vertical offset between text lines
            # Display tag ID
            cv2.putText(color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Display only the 1 most significant digit for pose_R and pose_t
            pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
            pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
            cv2.putText(color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.imshow("color", color)
        if cv2.waitKey(1) == 27:
          cv2.destroyAllWindows()
          return T

import config
cfg = config.Config()
# SIZE_OF_CALIBRATION_TAG = 5 #cm
import numpy as np
import os
import shutil
from datetime import datetime

def backup_calib_file():
    if not os.path.exists('calib.txt'):
        return
    if not os.path.exists('calib_log'):
        os.makedirs('calib_log')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy('calib.txt', f'calib_log/calib_{timestamp}.txt')


def derive_metrics_from_logs(log_dir, camera_type):
    x0_list = []
    y0_list = []
    z0_list = []

    for filename in os.listdir(log_dir):
        if filename.startswith("calib_") and filename.endswith(".txt"):
            with open(os.path.join(log_dir, filename), "r") as f:
                rsCamConfig = f.readline().strip().split(',')
                daiCamConfig = f.readline().strip().split(',')
                calib = np.loadtxt(f, delimiter=",")
            
            if camera_type == "rsCamToRobot":
                camToRobot = calib[:4, :]
            elif camera_type == "daiCamToRobot":
                camToRobot = calib[4:, :]
            else:
                continue

            x0_list.append(camToRobot[0, -1])
            y0_list.append(camToRobot[1, -1])
            z0_list.append(camToRobot[2, -1])

    if x0_list and y0_list and z0_list:
        x0_array = np.array(x0_list)
        y0_array = np.array(y0_list)
        z0_array = np.array(z0_list)
        
        mean_x0 = np.mean(x0_array)
        std_x0 = np.std(x0_array)
        mean_y0 = np.mean(y0_array)
        std_y0 = np.std(y0_array)
        mean_z0 = np.mean(z0_array)
        std_z0 = np.std(z0_array)

        print(f"{camera_type} Metrics:")
        print(f"x0 - Mean: {mean_x0}, Standard Deviation: {std_x0}")
        print(f"y0 - Mean: {mean_y0}, Standard Deviation: {std_y0}")
        print(f"z0 - Mean: {mean_z0}, Standard Deviation: {std_z0}")
    else:
        print(f"No valid data found for {camera_type}.")

def runCameraCalib(input="read"):
  np.set_printoptions(precision=2, suppress=True)
  if input == "calib_setup1": # dai and rs camera have common field of view and are looking at the same tag
    print("Running Camera Calibration")
    rsCam = RealSenseCamera(640,360)
    rsCamCalib = CalibrateCamera(rsCam)
    rsCamToRobot =   rsCamCalib.getCamera2Robot(tag_size=cfg.SIZE_OF_CALIBRATION_TAG, tag_id=0,viz=True)
    print("rsCamToRobot = \n", rsCamToRobot)

    # daiCam = DepthAICamera(640,360,object_detection=False)
    # daiCamCalib = CalibrateCamera(daiCam)
    # daiCamToRobot =   daiCamCalib.getCamera2Robot(tag_size=cfg.SIZE_OF_CALIBRATION_TAG, tag_id=0, viz=True)
    # print("daiCamToRobot = \n", daiCamToRobot)   
    backup_calib_file()
    with open("calib.txt", "w") as f:
        f.write(f"{rsCam.width},{rsCam.height}\n")
        # f.write(f"{daiCam.width},{daiCam.height}\n")
        np.savetxt(f, rsCamToRobot, delimiter=",")
        # np.savetxt(f, daiCamToRobot, delimiter=",")
        # Save RealSense camera intrinsics
        fx, fy, cx, cy, width, height, depth_scale = rsCam.getCameraIntrinsics()
        f.write(f"{fx},{fy},{cx},{cy},{width},{height},{depth_scale}\n")

  elif input == "read":
    # read the camera calibration matrices from the file "calib.txt"
    print("reading the camera calibration from file")
    with open("calib.txt", "r") as f:
        rsCamConfig = f.readline().strip().split(',')
        # daiCamConfig = f.readline().strip().split(',')
        rsCamWidth, rsCamHeight = int(rsCamConfig[0]), int(rsCamConfig[1])
        # daiCamWidth, daiCamHeight = int(daiCamConfig[0]), int(daiCamConfig[1])
        rsCamToRobot = np.loadtxt(f, max_rows=4, delimiter=",")
        # daiCamToRobot = np.loadtxt(f, max_rows=4, delimiter=",")
        intrinsicsRs = f.readline().strip().split(',')
        fx, fy, cx, cy, width, height, depth_scale = map(float, intrinsicsRs)
    print("rsCamToRobot = \n", rsCamToRobot)
    # print("daiCamToRobot = \n", daiCamToRobot)
    print(f"rsCam configuration: width={rsCamWidth}, height={rsCamHeight}")
    # print(f"daiCam configuration: width={daiCamWidth}, height={daiCamHeight}")   
    print(f"RealSense Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}, width={width}, height={height}, depth_scale={depth_scale}")
    derive_metrics_from_logs("calib_log", "rsCamToRobot")
    # derive_metrics_from_logs("calib_log", "daiCamToRobot")
  else:
    print("incorrect argument")
  
  return rsCamToRobot#, daiCamToRobot

def read_calibration_data():
    try:
        with open("calib.txt", "r") as f:
            rsCamConfig = f.readline().strip().split(',')
            # daiCamConfig = f.readline().strip().split(',')
            rsCamWidth, rsCamHeight = int(rsCamConfig[0]), int(rsCamConfig[1])
            # daiCamWidth, daiCamHeight = int(daiCamConfig[0]), int(daiCamConfig[1])
            rsCamToRobot = np.loadtxt(f, max_rows=4, delimiter=",")
            # daiCamToRobot = np.loadtxt(f, max_rows=4, delimiter=",")
            intrinsicsRs = f.readline().strip().split(',')
            fx, fy, cx, cy, width, height, depth_scale = map(float, intrinsicsRs)
        return {
            'rsCamWidth': rsCamWidth,
            'rsCamHeight': rsCamHeight,
            # 'daiCamWidth': daiCamWidth,
            # 'daiCamHeight': daiCamHeight,
            'rsCamToRobot': rsCamToRobot,
            # 'daiCamToRobot': daiCamToRobot,
            'intrinsicsRs': {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height,
                'depth_scale': depth_scale
            }
        }
    except FileNotFoundError:
        print("Error: The file 'calib.txt' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    

import argparse
if __name__ == "__main__":
  np.set_printoptions(precision=2, suppress=True)
  parser = argparse.ArgumentParser(description="Camera scripts which includes Calibration, and other functions like object detection")
  parser.add_argument("command", choices=["calib_setup1", "read"], help="Command to run")
  args = parser.parse_args()
  runCameraCalib(args.command)