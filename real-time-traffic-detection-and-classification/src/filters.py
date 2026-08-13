"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    filters.py

    This file is used to filter out objects that are too far away or too close
    to the camera, not moving, or moving in the opposite direction then
    specified.
"""
import configs

def is_too_far(box_height):
    """
        Checks if an object is too far away.

        box_height: int - Height of the box in pixels.
        
        returns True if the box is shorter than the minimum.
    """
    return box_height < configs.MIN_BOX_HEIGHT_PX

def is_too_close(box_height):
    """
        Checks if an object is too close.

        box_height: int - Height of the box in pixels.
        
        returns True if the box is larger than the maximum.
    """
    return box_height > configs.MAX_BOX_HEIGHT_PX

def is_stationary(state):
    """
        Checks if the object is not moving.

        state: Tracker - The tracker history for this object.

        returns True if it moves less then the designated minimum.
    """
    return state.distance_moved() < configs.STATIONARY_DISPLACEMENT_PX

def direction_label(state, camera_direction):
    """
        Converts the object left/right motion into a compas direction.

        state: Tracker - The tracker history for this object.
        camera_direction: str - The direction the camera faces 

        returns the compass direction the object is traveling or "" if unknown.
    """
    traffic_dir_str = ""

    # The two compass directions traffic can move for this camera facing.
    traffic_directions = configs.DIRECTIONS[camera_direction]

    if state.direction() ==  "left_to_right":
        traffic_dir_str = traffic_directions[0]

    elif state.direction() == "right_to_left":
        traffic_dir_str = traffic_directions[1]

    return traffic_dir_str

def matches_direction(state, wanted_direction):
    """
        Checks if the object is moving the way we asked to count.

        state: Tracker - The tracks history for this object.
        wanted_direction: str - "left_to_right", "right_to_left", or "both".

        returns True if the direction matches or is both.
    """
    if wanted_direction == "both":
        return True
        
    return state.direction() == wanted_direction
