from cortano import RealsenseCamera, VexV5

from planning import Planning

if __name__ == "__main__":
  camera = RealsenseCamera()
  robot = VexV5()
  planning = Planning()


  class State(Enum):
    CLAW_EMPTY = auto()
    TO_COKE_CAN = auto()
    PICKUP_COKE = auto()
    TO_GOAL_AREA = auto()
    DROP_CAN = auto()

  while robot.running():
    keys = robot.controller.keys
    y = keys["w"] - keys["s"] 
    x = keys["d"] - keys["a"]
    robot.motor[0] = (y + x) * 50
    robot.motor[9] = (y - x) * 50
    robot.motor[7] = (keys["p"] - keys["l"]) * 100
    robot.motor[2] = (keys["o"] - keys["k"]) * 100

    state = State.CLAW_EMPTY
    # refresh map all the time?

    if state == State.CLAW_EMPTY:
      # refresh map
      # is this state same as to coke can?
      state = State.TO_COKE_CAN

    elif state == State.TO_COKE_CAN:
      # refresh map
      if planning.get_plan_to_nearest_enemy_coke():
      # one robot gets to the nearest coke, change state to PICKUP_COKE
        state = State.PICKUP_COKE

    elif state == State.PICKUP_COKE:
      if planning.get_pickup_coke_can_plan():
        state = State.TO_GOAL_AREA

    elif state == State.TO_GOAL_AREA:
      if planning.get_plan_to_goal_area():
        state = State.DROP_CAN

    elif state == State.DROP_CAN:
      # maybe this state can be combined with to goal area and can is auto dropped when in goal area
      if planning.get_release_and_knock_over_can_plan():
        state = State.TO_COKE_CAN
