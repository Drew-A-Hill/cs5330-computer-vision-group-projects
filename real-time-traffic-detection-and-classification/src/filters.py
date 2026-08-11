import configs


def is_too_far(box_h):
    return box_h < configs.MIN_BOX_HEIGHT_PX

def is_stationary(state):
    return state.displacement() < configs.STATIONARY_DISPLACEMENT_PX

def direction_label(state, camera_direction):
    traffic_dir_str = ""
    traffic_directions = configs.DIRECTIONS[camera_direction]

    if state.direction() ==  "left_to_right":
        traffic_dir_str = traffic_directions[0]
    elif state.direction() == "right_to_left":
        traffic_dir_str = traffic_directions[1]

    return traffic_dir_str

def matches_direction(state, wanted_direction):
    if wanted_direction == "both":
        return True
    return state.direction() == wanted_direction
