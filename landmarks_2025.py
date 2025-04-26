import numpy as np

feet2cm = 30.48
heightOfTag = 28.575  # 11 1/4 inches in cm

map_apriltag_poses_game = {}

'''
Tag positions:
Top (0,1): At (1,4) and (-1,4)
Left (2-5): At (-2,3), (-2,1), (-2,-1), (-2,-3)
Bottom (6,7): At (-1,-4) and (1,-4)
Right (8-11): At (2,-3), (2,-1), (2,1), (2,3)

Tag orientations:
Y-axis consistently points into field (-Z of field) [0, 0, -1]
X and Z axes follow right-hand rule for each tag's facing direction
Proper counter-clockwise ordering starting from top right

'''
# Top side landmarks (Landmarks 0, 1) - facing south
for idx, i in enumerate(range(0, 2)):
    T = np.identity(4)
    T[:3, 0] = [1, 0, 0]    # X points right
    T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
    T[:3, 2] = [0, 1, 0]   # Z points south
    x = (3 - idx * 2) * feet2cm  # Places tag 0 at (1,4) and tag 1 at (-1,4)
    y = 4 * feet2cm  # Top edge
    T[:3, 3] = [x, y, heightOfTag]
    map_apriltag_poses_game[i] = T

# Left side landmarks (Landmarks 2, 3, 4, 5) - facing east
for idx, i in enumerate(range(2, 6)):
    T = np.identity(4)
    T[:3, 0] = [0, 1, 0]    # X points up
    T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
    T[:3, 2] = [-1, 0, 0]    # Z points east
    x = 0 * feet2cm  # Left edge
    y = (7 - (idx * 2)) * feet2cm  # Places tags at y=3,1,-1,-3 starting from top
    T[:3, 3] = [x, y, heightOfTag]
    map_apriltag_poses_game[i] = T

# Bottom side landmarks (Landmarks 6, 7) - facing north
for idx, i in enumerate(range(6, 8)):
    T = np.identity(4)
    T[:3, 0] = [-1, 0, 0]   # X points left
    T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
    T[:3, 2] = [0, -1, 0]    # Z points north
    x = 0 * feet2cm  # Places tags at (-1,-4) and (1,-4)
    y = 1 + (idx*2) * feet2cm  # Bottom edge
    T[:3, 3] = [x, y, heightOfTag]
    map_apriltag_poses_game[i] = T

# Right side landmarks (Landmarks 8, 9, 10, 11) - facing west
for idx, i in enumerate(range(8, 12)):
    T = np.identity(4)
    T[:3, 0] = [0, -1, 0]   # X points down
    T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
    T[:3, 2] = [1, 0, 0]   # Z points west
    x = 4 * feet2cm  # Right edge
    y = (1 + idx * 2) * feet2cm  # Places tags at y=-3,-1,1,3 starting from bottom
    T[:3, 3] = [x, y, heightOfTag]
    map_apriltag_poses_game[i] = T


# ... existing code ...

def get_apriltag_poses(field_setup='home'):
    map_apriltag_poses = {}

    if field_setup == 'game':
        # Top side landmarks (Landmarks 0, 1) - facing south
        for idx, i in enumerate(range(0, 2)):
            T = np.identity(4)
            T[:3, 0] = [1, 0, 0]    # X points right
            T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
            T[:3, 2] = [0, -1, 0]   # Z points south
            x = (1 - idx * 2) * feet2cm  # Places tag 0 at (1,4) and tag 1 at (-1,4)
            y = 4 * feet2cm  # Top edge
            T[:3, 3] = [x, y, heightOfTag]
            map_apriltag_poses[i] = T

        # Left side landmarks (Landmarks 2, 3, 4, 5) - facing east
        for idx, i in enumerate(range(2, 6)):
            T = np.identity(4)
            T[:3, 0] = [0, 1, 0]    # X points up
            T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
            T[:3, 2] = [1, 0, 0]    # Z points east
            x = -2 * feet2cm  # Left edge
            y = (3 - (idx * 2)) * feet2cm  # Places tags at y=3,1,-1,-3 starting from top
            T[:3, 3] = [x, y, heightOfTag]
            map_apriltag_poses[i] = T

        # Bottom side landmarks (Landmarks 6, 7) - facing north
        for idx, i in enumerate(range(6, 8)):
            T = np.identity(4)
            T[:3, 0] = [-1, 0, 0]   # X points left
            T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
            T[:3, 2] = [0, 1, 0]    # Z points north
            x = (-1 + idx * 2) * feet2cm  # Places tags at (-1,-4) and (1,-4)
            y = -4 * feet2cm  # Bottom edge
            T[:3, 3] = [x, y, heightOfTag]
            map_apriltag_poses[i] = T

        # Right side landmarks (Landmarks 8, 9, 10, 11) - facing west
        for idx, i in enumerate(range(8, 12)):
            T = np.identity(4)
            T[:3, 0] = [0, -1, 0]   # X points down
            T[:3, 1] = [0, 0, -1]   # Y points into field (-Z of field)
            T[:3, 2] = [-1, 0, 0]   # Z points west
            x = 2 * feet2cm  # Right edge
            y = (-3 + idx * 2) * feet2cm  # Places tags at y=-3,-1,1,3 starting from bottom
            T[:3, 3] = [x, y, heightOfTag]
            map_apriltag_poses[i] = T

    elif field_setup == 'home':
        map_apriltag_poses = {}
        #field origin is the corner of the room
        # tags 0, 1 are on -X axis
        # tag 2,3,4 are on -Y axis

        feet2inch = 12
        heightOfTag = 10.25 # 10 1/4 inches 

        # **Tag 0** - Top left at (-3 ft, 0 ft)
        T0 = np.identity(4)
        T0[:3, 0] = [1,  0,  0]    # X-axis points right
        T0[:3, 1] = [0,  0, -1]    # Y-axis points into field (-Z of field)
        T0[:3, 2] = [0,  1,  0]    # Z-axis points down
        T0[:3, 3] = [-3*feet2inch, 0, heightOfTag]
        map_apriltag_poses[0] = T0

        # **Tag 1** - Top right at (-1 ft, 0 ft)
        T1 = np.identity(4)
        T1[:3, 0] = [1,  0,  0]    # X-axis points right
        T1[:3, 1] = [0,  0, -1]    # Y-axis points into field (-Z of field)
        T1[:3, 2] = [0,  1,  0]    # Z-axis points down
        T1[:3, 3] = [-1*feet2inch, 0, heightOfTag]
        map_apriltag_poses[1] = T1

        # **Tag 2** - Right side top at (0 ft, -1 ft)
        T2 = np.identity(4)
        T2[:3, 0] = [0, -1,  0]    # X-axis points down
        T2[:3, 1] = [0,  0, -1]    # Y-axis points into field (-Z of field)
        T2[:3, 2] = [1,  0,  0]    # Z-axis points left
        T2[:3, 3] = [0, -1*feet2inch, heightOfTag]
        map_apriltag_poses[2] = T2

        # **Tag 3** - Right side middle at (0 ft, -3 ft)
        T3 = np.identity(4)
        T3[:3, 0] = [0, -1,  0]    # X-axis points down
        T3[:3, 1] = [0,  0, -1]    # Y-axis points into field (-Z of field)
        T3[:3, 2] = [1,  0,  0]    # Z-axis points left
        T3[:3, 3] = [0, -3*feet2inch, heightOfTag]
        map_apriltag_poses[3] = T3

        # **Tag 4** - Right side bottom at (0 ft, -5 ft)
        T4 = np.identity(4)
        T4[:3, 0] = [0, -1,  0]    # X-axis points down
        T4[:3, 1] = [0,  0, -1]    # Y-axis points into field (-Z of field)
        T4[:3, 2] = [1,  0,  0]    # Z-axis points left
        T4[:3, 3] = [0, -5*feet2inch, heightOfTag]
        map_apriltag_poses[4] = T4

    else:
        raise ValueError("Invalid field_setup. Choose 'home' or 'game'.")

    return map_apriltag_poses

# Update the main apriltag poses variable
# map_apriltag_poses = map_apriltag_poses_game #get_apriltag_poses(field_setup='game')
map_apriltag_poses = get_apriltag_poses(field_setup='home')

if __name__ == "__main__":
    # print the poses
    for i, pose in map_apriltag_poses.items():
        print(f"Tag {i}:")
        print(pose)
        print("\n")
