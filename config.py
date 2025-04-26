import math
class Config:
  TAG_POLICY = "HIGHEST_CONFIDENCE"
  TAG_POLICY = "FIRST"
  TAG_DECISION_MARGIN_THRESHOLD = 150
  TAG_POSE_ERROR_THRESHOLD = 0.01
  TAG_SIZE_3IN = 3 #7.62 # cm
  TAG_SIZE_6IN = 6 #15.24 # cm
  GEN_DEBUG_IMAGES = False
  SIZE_OF_CALIBRATION_TAG = 5 # inch or 12.7 cm
  CALIB_PATH = "/home/nvidia/wsp/clawbot2025/CortexNanoBridge/jetson_nano/calib.txt"
  # FIELD = "GAME" 
  FIELD = "HOME" 
  # FIELD = "BEDROOM" 
  ROBOT_LENGTH = 17.7165 #45 # cm
  ROBOT_WIDTH = 16.5354 # cm
  ROBOT_HEIGHT = 11.811 # cm
  ROBOT_RADIUS = math.sqrt(ROBOT_LENGTH**2 + ROBOT_WIDTH**2) / 2
  WHEEL_RADIUS = 2 #5.08 # in cm , 2 inch
  WHEEL_CIRCUMFERENCE = 2 * math.pi * WHEEL_RADIUS
  WHEEL_DISTANCE = 11.4173 #29 # cm
  WHEEL_DISTANCE_SIDE = 8.26772 # cm
  
  
  # FIELD = "GAME"
  FIELD_WIDTH = 48 # cm
  FIELD_HEIGHT = 48*2 # cm
  CLAW_LENGTH = 5.5 # cm
  CREATE_MASK = False
  MASK_PATH = "/home/nvidia/wsp/clawbot/Cortano/mask.png"

  LIVE_BALL_DEBUG = True

