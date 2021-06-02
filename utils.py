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


def circle_detection(img, display=False):

    output = img.copy()
    minDist = 100
    param1 = 30 #500
    param2 = 50 #200 #smaller value-> more false circles
    minRadius = 5
    maxRadius = 100 #10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # ensure at least some circles were found
    if circles is not None:
        print("CIRCLE DETECTED!!")
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        if display:
            # show the output image
            cv2.imshow("Circle", output)
            cv2.waitKey(10)
    
    return circles

def waypoints_adder_v2(waypoints, general_index, next_waypoint_distance, ego_state):    
    print("Aggiungo un waypoint al centro")
    x,y = from_global_to_local_frame(ego_state, waypoints[general_index][:2])
    
    added_waypoint = [[0,0,0]]
    x_l = x - next_waypoint_distance
    y_l = y
    print("Previous waypoints:", waypoints)
    x_g, y_g = from_local_to_global_frame(ego_state,[x_l,y_l])
    added_waypoint[0][0] = x_g
    added_waypoint[0][1] = y_g
    added_waypoint[0][2] = waypoints[general_index][2]
    temp = waypoints[general_index:]
    waypoints.resize((len(waypoints)+1, 3), refcheck = False )
    waypoints[general_index:general_index+1] = np.array(added_waypoint)
    waypoints[general_index+1:] = temp  

    print("Added waypoint: ", added_waypoint)
 
    print(waypoints)


'''ADDING WAYPOINTS'''
def waypoints_adder(waypoints, closest_index, goal_index, sampling_rate):
    print("Previous wp: ", waypoints)
    print("closest_index: ", closest_index)
    print("Goal index: ", goal_index)
    
    if(abs(waypoints[closest_index][0]-waypoints[goal_index][0])>0 or abs(waypoints[closest_index][1]-waypoints[goal_index][1])>0):
        print("waypoint corrente", waypoints[closest_index])
        step = (abs(waypoints[closest_index][0]-waypoints[goal_index][0]))/(sampling_rate+1)
        print("STEP IS: {}".format(step))
        added_waypoint = []
        for el in range(0,sampling_rate):
            added_waypoint.append([0,0,0])
        if(abs(waypoints[closest_index][1]-waypoints[goal_index][1])<0.1):
            direction = 0
            if(waypoints[closest_index][0]-waypoints[goal_index][0]>0):
                direction = -1
            else:
                direction = 1
            
            for el in range(0,sampling_rate):
                added_waypoint[el][0] = waypoints[closest_index][0] + direction*(el+1)*step
                added_waypoint[el][1] = waypoints[closest_index][1]
                added_waypoint[el][2] = waypoints[closest_index][2]

            print("added corrected {} \n".format(added_waypoint))    
            if(direction == -1):
                added_waypoint.reverse()
            temp = waypoints[closest_index+1:]
            waypoints.resize((len(waypoints)+sampling_rate, 3), refcheck = False )
            print(waypoints.shape)
            waypoints[closest_index+1:closest_index+sampling_rate+1] = np.array(added_waypoint)[::-1]
            waypoints[closest_index + sampling_rate+1:] = temp  

            print(waypoints)
            
        
        elif(abs(waypoints[closest_index][0]-waypoints[goal_index][0])<0.1):
            direction = 0
            if(waypoints[closest_index][1]-waypoints[goal_index][1]>0):
                direction = -1
            else:
                direction = 1
            
            for el in range(0,sampling_rate):
                added_waypoint[el][1] = waypoints[closest_index][1] + direction*(el+1)*step
                added_waypoint[el][0] = waypoints[closest_index][0]
                added_waypoint[el][2] = waypoints[closest_index][2]

            print("added corrected {} \n".format(added_waypoint))    
            if(direction == -1):
                added_waypoint.reverse()
            temp = waypoints[closest_index+1:]
            waypoints.resize((len(waypoints)+sampling_rate+1, 3), refcheck = False )
            print(waypoints.shape)
            waypoints[closest_index+1:closest_index+sampling_rate+1] = np.array(added_waypoint)[::-1]
            waypoints[closest_index+sampling_rate+1:] = temp  

            print(waypoints)


def waypoints_adder_in_prova(waypoints, closest_index, goal_index, sampling_rate):
    print("Previous wp: ", waypoints)
    print("Goal index: ", goal_index)
    
    if(abs(waypoints[closest_index][0]-waypoints[goal_index][0])>0 or abs(waypoints[closest_index][1]-waypoints[goal_index][1])>0):
        print("waypoint corrente", waypoints[closest_index])
        step = 0
        if(abs(waypoints[closest_index][1]-waypoints[goal_index][1])<0.1):
            step = (abs(waypoints[closest_index][0]-waypoints[goal_index][0]))/(sampling_rate+1)
        elif(abs(waypoints[closest_index][0]-waypoints[goal_index][0])<0.1) :
            step = (abs(waypoints[closest_index][1]-waypoints[goal_index][1]))/(sampling_rate+1)
        print("STEP IS: {}".format(step))
        added_waypoint = []
        for el in range(0,sampling_rate):
            added_waypoint.append([0,0,0])
        if(abs(waypoints[closest_index][1]-waypoints[goal_index][1])<0.1):
            direction = 0
            if(waypoints[closest_index][0]-waypoints[goal_index][0]>0):
                direction = -1
            else:
                direction = 1
            
            for el in range(0,sampling_rate):
                added_waypoint[el][0] = waypoints[closest_index][0] + direction*(el+1)*step
                added_waypoint[el][1] = waypoints[closest_index][1]
                added_waypoint[el][2] = waypoints[closest_index][2]
 
            print("added corrected {} \n".format(added_waypoint))    
            if(direction == -1):
                added_waypoint.reverse()
            temp = waypoints[closest_index+1:]
            waypoints.resize((len(waypoints)+sampling_rate, 3), refcheck = False )
            print(waypoints.shape)
            waypoints[closest_index+1:closest_index+sampling_rate+1] = np.array(added_waypoint)[::-1]
            waypoints[closest_index + sampling_rate+1:] = temp  
 
            print(waypoints)
            
        
        elif(abs(waypoints[closest_index][0]-waypoints[goal_index][0])<0.1):
            direction = 0
            if(waypoints[closest_index][1]-waypoints[goal_index][1]>0):
                direction = -1
            else:
                direction = 1
            
            for el in range(0,sampling_rate):
                added_waypoint[el][1] = waypoints[closest_index][1] + direction*(el+1)*step
                added_waypoint[el][0] = waypoints[closest_index][0]
                added_waypoint[el][2] = waypoints[closest_index][2]
 
            print("added corrected {} \n".format(added_waypoint))    
            if(direction == -1):
                added_waypoint.reverse()
            temp = waypoints[closest_index+1:]
            waypoints.resize((len(waypoints)+sampling_rate+1, 3), refcheck = False )
            print(waypoints.shape)
            waypoints[closest_index+1:closest_index+sampling_rate+1] = np.array(added_waypoint)[::-1]
            waypoints[closest_index+sampling_rate+1:] = temp  
 
            print(waypoints)



def waypoint_adder_ahead(waypoints, closest_index , ego_state):    
    
    print("Aggiungo un waypoint avanti")
    print("Previous waypoints", waypoints)

    x1,y1 = from_global_to_local_frame(ego_state, waypoints[closest_index][:2])


    x2,y2 = from_global_to_local_frame(ego_state, waypoints[closest_index+1][:2])
   

    added_distance =  abs(x1-x2)//2



    added_waypoint = [[0,0,0]]

    x_l = x1 + added_distance

    y_l = y1

 

    x_g, y_g = from_local_to_global_frame(ego_state,[x_l,y_l])

    added_waypoint[0][0] = x_g

    added_waypoint[0][1] = y_g

    added_waypoint[0][2] = waypoints[closest_index][2]

    temp = waypoints[closest_index:]

    waypoints.resize((len(waypoints)+1, 3), refcheck = False )

    print(waypoints.shape)

    waypoints[closest_index:closest_index+1] = np.array(added_waypoint)

    waypoints[closest_index+1:] = temp  

    print("Added waypoint: ", added_waypoint)

    print(waypoints)

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
                x_g,y_g = from_local_to_global_frame(ego_state, [next_waypoint_distance, waypoints[closest_index]])
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
"""

def turn_to_intersection(waypoints, intersection_point, ego_state):
    
    d = np.Inf
    min_idx = None
    for i in range(len(waypoints)):
        dist = np.linalg.norm(intersection_point[:2] - waypoints[i][:2])
        if dist < d:
            dist=d
            min_idx = i

    if min_idx == 0 or min_idx == len(waypoints)-1: #Se i punti più vicini all'intersezione sono l'inizio o la fine probabilmante alla prossima intersezione non dobbiamo girare
        return False

    L=5 if len(waypoints)>10 else len(waypoints)//2 -1

 
    #Prendi un wp a 20m daldove stai mo
    previous = waypoints[min_idx-L]
    next = waypoints[min_idx+L]
    if pointOnSegment(previous, waypoints[min_idx], next):
        return False
    else:
        return True
    
    local_prev = from_global_to_local_frame(ego_state, previous[:2])
    local_next = from_global_to_local_frame(ego_state, next[:2])
    if abs(local_next[1] - local_prev[1]) < 0.1: #Le y differiscono di un valore piccolo
        return False
    return True
"""

def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False

        




        


        