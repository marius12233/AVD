import cv2
import numpy as np
from math import cos, sin, pi,tan

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def from_local_to_global_frame(ego_state, vehicle_frame_pos):
    x,y = vehicle_frame_pos

    
    x_global = ego_state[0] + x*cos(ego_state[2]) - \
                                    y*sin(ego_state[2])
    y_global = ego_state[1] + x*sin(ego_state[2]) + \
                                    y*cos(ego_state[2])
    return x_global, y_global


def from_global_to_local_frame(ego_state, global_frame_pos):
    x_global, y_global = global_frame_pos
    x_g = x_global - ego_state[0]
    y_g = y_global - ego_state[1]
    x_l =  x_g*cos(ego_state[2]) + \
                                    y_g*sin(ego_state[2])

    y_l =  - x_g*sin(ego_state[2]) + \
                                    y_g*cos(ego_state[2])
    return x_l, y_l