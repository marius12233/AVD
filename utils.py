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


def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False

        
# Transform the obstacle with its boundary point in the global frame
def obstacle_to_world(location, dimensions, orientation):
    box_pts = []

    x,y = location[:2]

    yaw = orientation[0] * pi / 180

    xrad = dimensions.x
    yrad = dimensions.y
    zrad = dimensions.z

    # Border points in the obstacle frame
    cpos = np.array([
            [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
            [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
    
    # Rotation of the obstacle
    rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]])
    
    # Location of the obstacle in the world frame
    cpos_shift = np.array([
            [x, x, x, x, x, x, x, x],
            [y, y, y, y, y, y, y, y]])
    
    cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

    for j in range(cpos.shape[1]):
        box_pts.append([cpos[0,j], cpos[1,j]])
    
    return box_pts


#Offset will be used only for traffic lights
def waypoint_precise_adder(waypoints, next_waypoint_distance, closest_index, goal_index, tolerance, ego_state, offset=0.1):
    added_waypoint = [[0,0,0]]
    heading_index = None 
    minor=[np.inf ,np.inf]
    minor_index=None
    local=None 
    print("Closest index: ", closest_index)
    print("Goal index: ", goal_index)

    #Se mi sto fermando a un waypoint ma non sono ancora fermo e passo quel waypoint, 
    #il closest index diventerà  maggiore del goal_index. In questo caso 
    #restituisco il goal index
    if closest_index > goal_index:
        return goal_index

    if goal_index < len(waypoints) -1:
        goal_index+=1
    if closest_index > 0:
        closest_index-=1

    for i in range(closest_index, goal_index):
        local=from_global_to_local_frame(ego_state,waypoints[i][:2])
        if local[0] < next_waypoint_distance :
            minor=local
            minor_index=i
            continue

        elif local[0] >= next_waypoint_distance :
            if tolerance is not None:
                if abs(next_waypoint_distance-minor[0])<=tolerance:
                    print("Minor waypoint used: " ,waypoints[minor_index][:2])
                    print("minor distance ", minor[0])
                    return minor_index
                elif abs(local[0]-next_waypoint_distance)<=tolerance:
                    print("Major waypoint used: " ,waypoints[i][:2])
                    print("major distance ", local[0])
                    return i
            
                heading_index=i
            else:
                heading_index=i

        if heading_index is not None:
            break
    
    print("Heading index: ", heading_index)

    if heading_index is None:
        return goal_index

    position_index = heading_index - 1 if heading_index > 0 else heading_index #Proviamo a mettere la posizione del waypoint precedente
    local=from_global_to_local_frame(ego_state,waypoints[position_index][:2])
    x_g,y_g = from_local_to_global_frame(ego_state, [next_waypoint_distance,local[1]+offset]) 
    added_waypoint[0][0] = x_g
    added_waypoint[0][1] = y_g
    added_waypoint[0][2] = waypoints[heading_index][2]
    
    
    waypoints.resize((len(waypoints)+1, 3), refcheck = False )    

    waypoints[heading_index+1:] = waypoints[heading_index:-1]
    waypoints[heading_index] = np.array(added_waypoint)
    

    return heading_index

        