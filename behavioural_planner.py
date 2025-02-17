#!/usr/bin/env python3
from traffic_light import GREEN, RED, TrafficLight
from utils import from_global_to_local_frame
import numpy as np
import math
from utils import from_global_to_local_frame,waypoint_precise_adder
from road_manage import RoadHandler
# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
EMERGENCY_STOP = 4


# Stop speed threshold
STOP_THRESHOLD = 0.03
EMERGENCY_STOP_THRESHOLD = 0.002

#other costants
MAX_DIST_TO_STOP = 7
MIN_DIST_TO_STOP = 4
METER_TO_DECELERATE = 20
DIST_FROM_PEDESTRIAN = 4
STOP_FOR_PEDESTRIAN = 0
STOP_FOR_TL = 1
EGO_Y_EXTEND = 1
       

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._traffic_light:TrafficLight = None
        self._closest_pedestrian=None
        self._pedestrian_stopped_index = None
        self._stop_for = None #None if you will not stop for tl or ped, 0 for pedestrian, 1 for TL
        self._road_handler:RoadHandler = None

    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_follow_lead_vehicle_lookahead(self, follow_lead_vehicle_lookahead):
        self._follow_lead_vehicle_lookahead = follow_lead_vehicle_lookahead

    def set_traffic_light(self, traffic_light:TrafficLight):
        if self._traffic_light is None:
            self._traffic_light = traffic_light
    
    def set_road_handler(self, road_handler):
        self._road_handler = road_handler

    

    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """
        Handles state transitions and computes the goal state.
        """
        print("STATE: ", self._state)
        print("STOP FOR: ", self._stop_for)

        if self._state == FOLLOW_LANE:
            
            #if we are close to an intersection,we reduce the lookahead
            if self._road_handler.is_inrange_nearest_intersection(ego_state, ranging=15):
                if self._road_handler.is_turn():
                    self._lookahead=16

            print("Lookahead: ", self._lookahead)
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # Check traffic lights
            traffic_light_found_distance = self.distance_from_closest_traffic_light(ego_state, use_lookahead=True)

            #Check for pedestrian
            pedestrian_ahead_found_distance=self.distance_from_closest_pedestrian(ego_state)
            
            #if the distances are None, we set them to np.inf            
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if pedestrian_ahead_found_distance is None:
                pedestrian_ahead_found_distance=np.inf
            #if both are np.inf, next state will be follow lane and no changes to the goal state            
            if pedestrian_ahead_found_distance == np.inf and traffic_light_found_distance==np.inf: #Non mi sto fermando
                self._pedestrian_stopped_index=None
                self._stop_for = None
                return
            
            d_real = closed_loop_speed**2/5
            #if the pedestrian is closer then the tf light,we have to stop for pedestrian
            if pedestrian_ahead_found_distance < traffic_light_found_distance: 
                print("Add waypoint to pedestrian at distance: ", pedestrian_ahead_found_distance)              
                goal_index=waypoint_precise_adder(waypoints,pedestrian_ahead_found_distance, closest_index, goal_index, 0.1, ego_state, offset=0)
                self._stop_for = STOP_FOR_PEDESTRIAN
                self._pedestrian_stopped_index = self._closest_pedestrian["index"]
                

            #we have to stop for the tf light
            else: 
                
                #we add a waypoint for the tf light 
                #we check if starting to decelerate th car will stop  before the desired distance or
                #will go to slow because th waypoint is too far

                #we can say that the  dist_current_stop = -v^2/2*a,  with a = -2.5 at max
                #if dist_current_stop < dist_preferred_stop: return
                if traffic_light_found_distance > MAX_DIST_TO_STOP:
                    print("I want to stop at distance: ", traffic_light_found_distance)
                    print("I will stop at distance: ", d_real)
                    if d_real < traffic_light_found_distance - 1: 
                        print("I go too slowly to stop where I want")
                        return

                print("Add waypoint to traffic light")
                goal_index=waypoint_precise_adder(waypoints,traffic_light_found_distance,closest_index, goal_index,0.1,ego_state)    
                self._stop_for = STOP_FOR_TL

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            self._goal_state[2] = 0
            self._state = DECELERATE_TO_STOP



        elif self._state == DECELERATE_TO_STOP:
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self._goal_index
            
            #we have to check if there is a new pedestrian colser than the previous one. 
            #we can do that checking the distance from goal index            
            
            if self._stop_for==STOP_FOR_PEDESTRIAN:
                print("I'm stopping for the pedestrian!!")
                pedestrian_ahead_found_distance=self.distance_from_closest_pedestrian(ego_state)
                goal_dist = from_global_to_local_frame(ego_state, waypoints[goal_index][:2])[0]

                if pedestrian_ahead_found_distance is None:
                    pedestrian_ahead_found_distance=np.inf
                
                #in this situation the pedestrian can be the same as before (is coming up to our vehicle)
                #or is a new pedestrian.
                #For both the cases we try to add a new waypoint and , if is not possible to stop before this waypoint, 
                #we go in the EMERGENCY STOP state
                
                goal_distance = goal_dist - 0.1 if goal_dist > 0 else goal_dist + 0.1
                if pedestrian_ahead_found_distance < goal_distance:#goal_dist - 0.1: 
                    d_real = closed_loop_speed**2/5
                    
                    if self._pedestrian_stopped_index != self._closest_pedestrian["index"]:
                        goal_index=waypoint_precise_adder(waypoints,pedestrian_ahead_found_distance, closest_index, goal_index, 0.1, ego_state, offset=0)
                        self._pedestrian_stopped_index = self._closest_pedestrian["index"]

                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                    self._goal_state[2] = 0
                    self._state = DECELERATE_TO_STOP

                    if d_real > pedestrian_ahead_found_distance:
                        self._state = EMERGENCY_STOP
                #in this case the pedestrian is no more in our scope
                elif pedestrian_ahead_found_distance == np.Inf: 
                    self._state = FOLLOW_LANE
                    self._pedestrian_stopped_index=None
                    self._stop_for=None
            
            elif self._stop_for == STOP_FOR_TL: 
                #if we were stopping for the trafiic light, we have to check if there is a pedestrian before our goal index
                #that for sure was setted for the traffic light
                #as usually, if we have no time for a decelerate, we go in emergency stop
                print("I'm stopping for the traffic light!!")
                pedestrian_ahead_found_distance=self.distance_from_closest_pedestrian(ego_state)
                if pedestrian_ahead_found_distance is None:
                    self._pedestrian_stopped_index=None
                    pedestrian_ahead_found_distance=np.inf

                d_real = closed_loop_speed**2/5
                if pedestrian_ahead_found_distance < from_global_to_local_frame(ego_state, waypoints[goal_index][:2])[0]:
                    print("I was stopping for the traffic light -> I'm stopping for the pedestrian")
                    print("I want to stop at distance: ", pedestrian_ahead_found_distance)
                    print("I will stop at distance: ", d_real)

                    
                    self._pedestrian_stopped_index = self._closest_pedestrian["index"]
                    
                    self._stop_for=STOP_FOR_PEDESTRIAN

                    goal_index=waypoint_precise_adder(waypoints,pedestrian_ahead_found_distance, closest_index, goal_index, 0.1, ego_state, offset=0)

                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                    self._goal_state[2] = 0 

                    if d_real > pedestrian_ahead_found_distance:
                        self._state = EMERGENCY_STOP


                else: 
                    
                    #if no pedestrian are in the scope, we just check if the traffic light is green (we can go in follow lane)
                    traffic_light_found_distance = self.distance_from_closest_traffic_light( ego_state, use_lookahead=False)
                    if traffic_light_found_distance is  None: 
                        self._state = FOLLOW_LANE
                        self._stop_for=None   
                    else:
                        #if we cannot stop before, we go in emergency stop
                        if d_real > traffic_light_found_distance and traffic_light_found_distance <= MIN_DIST_TO_STOP + 1 : 
                            self._state = EMERGENCY_STOP

                    

            if abs(closed_loop_speed) <= STOP_THRESHOLD and self._state == DECELERATE_TO_STOP:
                self._state = STAY_STOPPED


        elif self._state == STAY_STOPPED:
            
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self._goal_index
            #we check if there is no more the pedestrian (we go follow lane)
            if self._stop_for == STOP_FOR_PEDESTRIAN: 
                pedestrian_ahead_found_distance=self.distance_from_closest_pedestrian(ego_state)
                if pedestrian_ahead_found_distance is None:
                    self._stop_for=None
                    self._state = FOLLOW_LANE
                    self._pedestrian_stopped_index=None
            #we check if the traffic light is green (we go follow lane)
            elif self._stop_for == STOP_FOR_TL:
                traffic_light_found_distance = self.distance_from_closest_traffic_light( ego_state, use_lookahead=False)
                if traffic_light_found_distance is None:
                    self._stop_for=None
                    self._state = FOLLOW_LANE

        
        elif self._state == EMERGENCY_STOP:

            if abs(closed_loop_speed) <= EMERGENCY_STOP_THRESHOLD:
                self._state = STAY_STOPPED


    def distance_from_closest_traffic_light(self,  ego_state, use_lookahead=True):
        """Returns the distance from the closest  traffic light if is red/yellow else None
        """
        if self._traffic_light is None:
            return None
        
        print("STATE Traffic light: is next: {}, color:{}"
        .format(self._traffic_light.is_next(), 
        "RED" if self._traffic_light._color else "GREEN"
        ))
        #if the current setted tf light is not close enough, return none
        if not self._traffic_light.is_next():
            return None
        #if the tf light is green return none
        color = self._traffic_light.get_color()
        if color == GREEN:
            return None
        s = np.array(self._traffic_light.get_position()[0:2])
        s_local = from_global_to_local_frame(ego_state, s) 
        #my lookahead is smaller than the distance from the tflight
        if use_lookahead and s_local[0] > self._lookahead: #Non sono arrivato col veicolo a guardare i waypoints nel range specificato (usato solo quando sto in follow lane)
            return None
        #set the distance to stop 6mt before the tflight
        preferred_distance = s_local[0] - MIN_DIST_TO_STOP - 1
        print("Preferred distance for traffic light: ", preferred_distance)
        return preferred_distance

        

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)
                
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car
        """
        if lead_car_position is None:
            self._follow_lead_vehicle = False
            return

        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            #if we are close to an intersection,we reduce the lookahead for the leading vehicle             
            if self._road_handler.is_inrange_nearest_intersection(ego_state, ranging=15):
                self._follow_lead_vehicle_lookahead=6
            # In this case, the car is too far away.   
            if lead_car_distance >  self._follow_lead_vehicle_lookahead:
                return
            lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), 
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.

            if np.dot(lead_car_delta_vector, 
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return
            self._follow_lead_vehicle = True

        else:
            
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            #if we are close to an intersection,we reduce the lookahead for the leading vehicle             
            if self._road_handler.is_inrange_nearest_intersection(ego_state, ranging=15):
                self._follow_lead_vehicle_lookahead=6
            #we will stop to following the leading car if it goes out of range (with a 5mt tolerance)            
            if lead_car_distance > self._follow_lead_vehicle_lookahead + 5:
                self._follow_lead_vehicle = False
                return
            
            
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 6:
                return
            
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False
        
    
    
    def check_for_vehicle(self,ego_state, vehicle_position,vehicle_bb, ):
        """
        Checks for all vehicle in a range that will be considered for the collision with the local planner paths 
        """
        prob_coll_vehicle=[]
        for i in range(len( vehicle_position )):
            obs_local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            if obs_local_pos[0]>0 and obs_local_pos[0] < 20 and obs_local_pos[1]<5 and obs_local_pos[1]>-5:
                prob_coll_vehicle.append(vehicle_bb[i])
        return prob_coll_vehicle
    

    
    def check_forward_closest_vehicle(self, ego_state, ego_orientation, vehicle_position, vehicle_rot):
        """
        Checks for a possible leading car
        """
        lead_car_idx=None
        lead_car_local_pos=None
        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x) 
        for i in range(len(vehicle_position)):
            vehicle_angle = math.atan2(vehicle_rot[i][1],vehicle_rot[i][0]) 
            local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            diff = abs(ego_angle - vehicle_angle)
            if diff > math.pi:
                diff = 2*math.pi - diff
            #if the right and the left boundaries are not found, them are setted to default
            left_bound = -5 if self._road_handler._boundaries[0] is None else self._road_handler._boundaries[0]
            right_bound = 5 if self._road_handler._boundaries[1] is None else self._road_handler._boundaries[1]
            #select only the car ahead and going in a similar direction (in a pi/4 angle range)
            if local_pos[0] >= 0: 
                if diff <= math.pi/4:
                     #we save only the closest one  
                    if (lead_car_idx is None or local_pos[0]<lead_car_local_pos[0]) and local_pos[1]>left_bound and local_pos[1]<right_bound :
                        lead_car_idx=i
                        lead_car_local_pos=local_pos
        return lead_car_idx

    def check_for_closest_pedestrian(self,ego_state,ego_orientation,pedestrian_position,pedestrian_rot, pedestrian_pixels):
        """
        Checks for all possible future colliding pedestrian, and sets just the closest one
        """
        local_pos_closest=np.inf
        closest_ped_idx=None
        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x)
        lookahead_dist=16
        
        for i in range(len( pedestrian_position )):

            local_pos=from_global_to_local_frame(ego_state,pedestrian_position[i])

            #Filter out pedestrian not on road
            if self._closest_pedestrian is None:
                x,y = pedestrian_pixels[i]
                y = 416 - y
                pedestrian_out_of_range = self._road_handler.is_on_sidewalk(x,y)
                if pedestrian_out_of_range:
                    continue
                


            pedestrian_angle = math.atan2(pedestrian_rot[i][1],pedestrian_rot[i][0])
            #if we are close to an intersection,we reduce the lookahead for the  pedestrian            
            if self._road_handler.is_inrange_nearest_intersection(ego_state, ranging=15):
                if self._road_handler.is_turn():
                    lookahead_dist=8
            
            diff = abs(ego_angle - pedestrian_angle)
            if diff > math.pi:
                diff = 2*math.pi - diff
                 
            #if the right and the left boundaries are not found, them are setted to default            
            left_bound = -5 if self._road_handler._boundaries[0] is None else self._road_handler._boundaries[0]
            right_bound = 5 if self._road_handler._boundaries[1] is None else self._road_handler._boundaries[1]

            if local_pos[0]> 0 and local_pos[0] <lookahead_dist and local_pos[1]>left_bound and local_pos[1]<right_bound: 
                 
                if (diff > math.pi/4 and diff < 3.5*math.pi/4) or (local_pos[1]>-EGO_Y_EXTEND and local_pos[1]<EGO_Y_EXTEND): #Se l'orientamento è più o meno perpendicolare o il pedone sta in un range uguale alla largezza del veicolo prendilo come closest pedestrian
                    if local_pos[0] < local_pos_closest:
                        local_pos_closest=local_pos[0]
                        closest_ped_idx=i
                    
        if local_pos_closest == np.inf:
            self._closest_pedestrian= None
        else:
            #update the info for the closest pedestrian
            if self._closest_pedestrian is None:
                self._closest_pedestrian={}
                self._closest_pedestrian["pos"]=pedestrian_position[closest_ped_idx]
                self._closest_pedestrian["index"]=closest_ped_idx
            elif self._closest_pedestrian["index"]==closest_ped_idx:
                self._closest_pedestrian["pos"]=pedestrian_position[closest_ped_idx]
            else:
                self._closest_pedestrian["pos"]=pedestrian_position[closest_ped_idx]
                self._closest_pedestrian["index"]=closest_ped_idx
        print("Closest pedestrian: ", self._closest_pedestrian)
        print("Dist from closest: ", local_pos_closest)
        return
                       
    def distance_from_closest_pedestrian(self,ego_state):
        """
        Returns the distance from closest pedestrian, if is not None
        """
        if self._closest_pedestrian is None:
            return None

        closest_pedestrian_local = from_global_to_local_frame(ego_state, self._closest_pedestrian["pos"])
        #we will try to stop 4 mt before the pedestrian        
        return closest_pedestrian_local[0]-DIST_FROM_PEDESTRIAN
        
    
    def check_for_pedestrian(self,ego_state, pedestrian_position,pedestrian_bb):
        '''checks for all the pedestrian in a range in front of the vehicle and returns the bounding boxes
        '''        
        prob_coll_pedestrian=[]
        left_bound = -5 if self._road_handler._boundaries[0] is None else self._road_handler._boundaries[0]
        right_bound = 5 if self._road_handler._boundaries[1] is None else self._road_handler._boundaries[1]
        for i in range(len( pedestrian_position )):
            #if i m already stopping for that pedestrian, i discard him from the list
            if self._closest_pedestrian and i == self._closest_pedestrian["index"]:
                continue
            obs_local_pos=from_global_to_local_frame(ego_state,pedestrian_position[i])
            #if the pedestrian is in front of me in a setted range, i add his bb to the list            
            if obs_local_pos[0]>0 and obs_local_pos[0] < self._lookahead and obs_local_pos[1]<right_bound and obs_local_pos[1]>left_bound:
                prob_coll_pedestrian.append(pedestrian_bb[i])
        return prob_coll_pedestrian
                    





        
# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
