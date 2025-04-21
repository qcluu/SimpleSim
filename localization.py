class Localization:
    def get_robot_position(): # → (x,y,theta)
        return {
            "x": 0, 
            "y":0,
            "theta": 0
            }
    
    def get_nearest_coke_can_on_enemy_side(): # → (x,y,theta)
        # Returns relative distance (x,y,theta)
        return {
            "x": 5, 
            "y":5,
            "theta": 0
            }
    
    def get_goal_area_landmarks(): # → Struct/Tuple of (x,y,theta)
        return "landmark placeholder"
    
    def update_map(): # → refresh the map based on the current view of the board
        return "map update placeholder"