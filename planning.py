import time

from localization import Localization
from controller import Control

controller = Control()
localization = Localization()

class Planning():
    def get_plan_to_nearest_enemy_coke():
        robot_position_info = Localization.get_robot_position()
        coke_can_position_info = Localization.get_nearest_coke_can_on_enemy_side()
        controller.send_to_XY(coke_can_position_info.x, coke_can_position_info.y)

        # implement some kind of check to see if coke can is there 
        # compare the april tags of coke can and claw and figure out this distance
        # TODO: in person figure out how to calculate this distance, this is placeholder formula
        dx = coke_can_position_info.x - robot_position_info.x
        dy = coke_can_position_info.y - robot_position_info.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance < 0.1:
            return True
        else:
            return False
        
    def get_pickup_coke_can_plan():
        controller.close_arm()
        # robot needs to close the arm first and hold the close arm button?
        # maybe a time.sleep() or something so that the arm is not raised too quickly
        controller.raise_arm()

        # TODO: is there a way that the robot can check if the coke can is held? Like maybe if the claw closes all the way
        held = False
        if held == True:
            return True
        else:
            return False
    
    def get_plan_to_goal_area():
        robot_position_info = Localization.get_robot_position()
        landmarks = Localization.get_goal_area_landmarks()

        # TODO: what is the format of goal area landmarks and how to check if robot inside
        # return true if robot is inside
    
    def get_release_and_knock_over_can_plan():
        controller.open_arm() # hopefully can will drop and tip over
        
        # reset the robot arm
        controller.lower_arm()

        return True
