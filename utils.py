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


def waypoint_add_ahead_distance(waypoints, closest_index, goal_index, next_waypoint_distance, ego_state):
    print("WAYPOINT ADDER")
    if next_waypoint_distance <0 :
        print("WAYPOINT ADDER: distance <0")
        return closest_index
    added_waypoint = [[0,0,0]]
    heading_index = None #L'indice a cui devo inserire il waypoint
    x_l,y_l = from_global_to_local_frame(ego_state, waypoints[closest_index][:2])
    #if x_l < next_waypoint_distance and x_l > next_waypoint_distance :
    #    heading_index=closest_index
    #    print("WAYPOINT ADDER: sto restituendo closest")
    #    return heading_index
        
         #Il closest index sta più avanti di dove voglio fermarmi
        #Devo mettere un waypoint dietro il closest index
        #print("BEFORE CLOSEST")
    if x_l > next_waypoint_distance:
        x_g,y_g = from_local_to_global_frame(ego_state, [next_waypoint_distance, y_l])
        heading_index = closest_index
        print("WAYPOINT ADDER: heading_index = closest_index")

    elif x_l < next_waypoint_distance : #La distanza dal closest  index è minore rispetto alla distanza che hai messo
        #print("AFTER CLOSEST")
        #scorri i waypoints avanti fino a che non trovi uno con distanza maggiore. QDevi 
        #mettere il nuovo waypoint dietro questo
         #Waypoint che sta dopo quello che devo mettere
        print("AFTER CLOSEST")
        goal = goal_index
        if goal_index<len(waypoints)-1:
            goal+=1
        for i in range(closest_index, goal):
            x_l2,y_l2 = from_global_to_local_frame(ego_state, waypoints[i][:2])
            if x_l2 > next_waypoint_distance:
                x_g,y_g = from_local_to_global_frame(ego_state, [next_waypoint_distance,y_l])
                heading_index = i
                break

    if heading_index is None:
        print("WAYPOINT ADDER: heading_index is None")
        return goal_index
    
    
    added_waypoint[0][0] = x_g
    added_waypoint[0][1] = y_g
    added_waypoint[0][2] = waypoints[heading_index][2]
    temp = waypoints[heading_index:]
    waypoints.resize((len(waypoints)+1, 3), refcheck = False )
    waypoints[heading_index:heading_index+1] = np.array(added_waypoint)
    waypoints[heading_index+1:] = temp
    print("Added waypoint in :",waypoints[heading_index][:2])
    return heading_index 


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

def waypoint_precise_adder(waypoints, next_waypoint_distance,tolerance,ego_state):
    
    added_waypoint = [[0,0,0]]
    heading_index = None 
    minor=[np.inf ,np.inf]
    local=None
    for i in range(len(waypoints)):
        local=from_global_to_local_frame(ego_state,waypoints[i][:2])
        if local[0] < next_waypoint_distance:
            minor=local
            pass
        if tolerance is not None:
            if abs(next_waypoint_distance-minor[0])<=tolerance:
                print("Minor waypoint used: " ,waypoints[i-1][:2])
                print("minor distance ", minor[0])
                return i-1
            elif abs(local[0]-next_waypoint_distance)<=tolerance:
                print("Major waypoint used: " ,waypoints[i][:2])
                print("major distance ", local[0])
                return i
        heading_index=i
        break

    x_g,y_g = from_local_to_global_frame(ego_state, [next_waypoint_distance,local[1]])
    added_waypoint[0][0] = x_g
    added_waypoint[0][1] = y_g
    added_waypoint[0][2] = waypoints[heading_index][2]
    temp = waypoints[heading_index:]
    waypoints.resize((len(waypoints)+1, 3), refcheck = False )
    waypoints[heading_index:heading_index+1] = np.array(added_waypoint)
    waypoints[heading_index+1:] = temp
    print("Added waypoint in :",waypoints[heading_index][:2])
    return heading_index

        


        