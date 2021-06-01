#!/usr/bin/env python3
from traffic_light import GREEN, TrafficLight
from utils import from_global_to_local_frame
import numpy as np
import math
from utils import from_global_to_local_frame, waypoints_adder, waypoints_adder_v2, waypoints_adder_in_prova,waypoint_adder_ahead,waypoint_add_ahead_distance

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
DECELERATE_TO_INTERSECTION = 3
CHANGE_LANE_ON_LEFT=4
# Stop speed threshold
STOP_THRESHOLD = 0.03
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10
MAX_DIST_TO_STOP = 7
MIN_DIST_TO_STOP = 3
METER_TO_DECELERATE = 20
#Traffic light
RED=1
GREEN=0

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count_for_pedestrian= 0
        self._no_tl_found_counter = 0
        self._traffic_light:TrafficLight = None
        self._has_tl_changed_pos = False #Ci serve per dire che se il semaforo è quello di sempre mi risparmio di fare le operazioni
        self._desired_speed_intersection = 2
        self._next_intersection = None
        self._closest_pedestrian=None
        
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_follow_lead_vehicle_lookahead(self, follow_lead_vehicle_lookahead):
        self._follow_lead_vehicle_lookahead = follow_lead_vehicle_lookahead

    def set_traffic_light(self, traffic_light:TrafficLight):
        if self._traffic_light is None:
            self._traffic_light = traffic_light
    
    def set_next_intersection(self, next_intersection):
        self._next_intersection = next_intersection

    def set_pedestrian_on_lane(self,pedestrian_on_lane):
        self._pedestrian_on_lane=pedestrian_on_lane 
    def set_position_to_stop(self,position_to_stop):
        self._position_to_stop=position_to_stop
    def get_follow_lead_vehicle(self):
        return self._follow_lead_vehicle 
    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
       
        print("STATE: ", self._state)
        if self._state == FOLLOW_LANE:
            #print("FOLLOW_LANE")
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            #Check intersection
            
            
            goal_index, traffic_light_found = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            
            if traffic_light_found: 
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                print("ROSSO!!")
                #print("TL Waypoint: ", from_global_to_local_frame(ego_state, self._goal_state[:2]))
                self._state = DECELERATE_TO_STOP
            
            
            goal_index,try_to_stop=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)

            if try_to_stop:
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                print("Mi fermo tra: ",from_global_to_local_frame(ego_state,waypoints[goal_index][:2]))
                self._state = DECELERATE_TO_STOP
        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
           
            
            color = RED
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            
            goal_index,try_to_stop=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)
            
            if try_to_stop:
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                print("Mi fermo tra: ",from_global_to_local_frame(ego_state,waypoints[goal_index][:2]))
                self._state=DECELERATE_TO_STOP
            elif  self._traffic_light is not None:
                color = self._traffic_light.get_color()
                if  color == GREEN or not self._traffic_light.is_next():
                    print("SEMAFORO CAMBIATO: ROSSO -> VERDE")
                    self._state = FOLLOW_LANE
                



            if abs(closed_loop_speed) <= STOP_THRESHOLD and self._state != FOLLOW_LANE:
                self._state = STAY_STOPPED  
                    

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:
            tf_light=True
            color=GREEN

            if self._traffic_light is not None:
                color = self._traffic_light.get_color()
                if color == RED and self._traffic_light.is_next():
                    tf_light=False   
                if  color==GREEN and self._traffic_light.is_next():
                    self._state = FOLLOW_LANE
            
            
            
            
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            goal_index,try_to_stop=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)
            if try_to_stop:
                self._state=STAY_STOPPED
            elif tf_light:
                self._state=FOLLOW_LANE




        else:
            raise ValueError('Invalid state value.')
        '''
        elif self._state == DECELERATE_TO_INTERSECTION:
            #Se sto a 20 m dall'incrocio torno in FOLLOW LANE
            next_intersection = self._next_intersection
            next_intersection_local = from_global_to_local_frame(ego_state, next_intersection[:2])
            
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

                # Next, find the goal index that lies within the lookahead distance
                # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

            if next_intersection_local[0] <= METER_TO_DECELERATE: #Se stiamo meno di 20 m dal prossimo incrocio
                self._state = FOLLOW_LANE
            else:
                 # Check stop signs
                
                
                goal_index, traffic_light_found = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
                
                #if self._traffic_light is not None:
                #    print("In Follow Lane: tl, tl_found", self._traffic_light.get_pos(), self._traffic_light.get_color(), self._traffic_light.is_next(), self._traffic_light.has_changed,traffic_light_found)
                
                if traffic_light_found: 
                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                    self._goal_state[2] = 0
                    print("ROSSO!! Da dec. to int. a dec. to stop")
                    self._state = DECELERATE_TO_STOP
                
            goal_index,try_to_stop=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)

            if try_to_stop:
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                print("Mi fermo tra: ",from_global_to_local_frame(ego_state,waypoints[goal_index][:2]))
                self._state = DECELERATE_TO_STOP
        '''
        


    def check_for_next_intersection(self, waypoints, closest_index, goal_index, ego_state):
        """[summary]
        Se so che ho un'intersezione davanti vorrei arrivare all'intersezione con una velocità più bassa.
        Per fare ciò prendo tra i waypoints nel lookahead il waypoint più lontano dall'intersezione che ha una distanza 
        almeno di 25 mt da essa

        Args:
            waypoints ([type]): [description]
            closest_index ([type]): [description]
            goal_index ([type]): [description]
            ego_state ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self._next_intersection is None:
            return goal_index, False
        
        next_intersection_local = from_global_to_local_frame(ego_state, self._next_intersection[:2])
        #Prendere il waypoint con distanza minore o uguale della prossima intersezione -20
        #Controllare se il veicolo sta più lontano di 20m
        if next_intersection_local[0] < METER_TO_DECELERATE:
            return goal_index, False

        dist = -np.Inf
        wp_idx_min = None
        for i in range(closest_index, goal_index):
            wp = waypoints[i]
            wp_local = from_global_to_local_frame(ego_state, wp[:2])
            d = next_intersection_local[0] - wp_local[0]

            if d > METER_TO_DECELERATE:
                continue

            #dist_v = 20 + self._lookahead #distanza a cui deve stare il veicolo per cui inizio a prendermi i punti

            if d > dist:
                d = dist
                wp_idx_min = i
        
        #Per ora restituiamo wp idx min è True
        #Controllare se wp_min_idx è None, e in questo caso
        #aggiungere dei waypoint
        if wp_idx_min is not None:
            return wp_idx_min, True
        
        return goal_index, False



    

    
    def check_for_traffic_light(self, waypoints, closest_index, goal_index, ego_state):
        """Checks for a stop sign that is intervening the goal path.

        Checks for a stop sign that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a stop line), and a
        boolean flag indicating if a stop sign obstruction was found.
        
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
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
                goal_index (current): Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        variables to set:
            [goal_index (updated), stop_sign_found]: 
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                stop_sign_found: Boolean flag for whether a stop sign was found or not
        """

        if self._traffic_light is None:
            return goal_index, False
        
        #Se il semaforo corrente non è più il prossimo dici che non c'è
        if not self._traffic_light.is_next():
            return goal_index, False

        #Col verde non facciamo niente
        color = self._traffic_light.get_color()
        if color == GREEN:
            print("SEMAFORO VERDE!")
            return goal_index, False
        

        s = np.array(self._traffic_light.get_pos()[0:2])
        s_local = from_global_to_local_frame(ego_state, s)
        
        if s_local[0] > self._lookahead: #Non sono arrivato col veicolo a guardare i waypoints nel range specificato
            return goal_index, False

        min_idx = None
        min_dist = np.Inf
        if self._traffic_light.has_changed or self._traffic_light._changed_color:
            #SCelgo di fermarmi a distanza 6 dal semaforo
            #distanza dal semaforo
            print("CHECK FOR TRAFFIC LIGHT")

            preferred_distance = s_local[0]-MIN_DIST_TO_STOP-1
            min_idx,bl = waypoint_add_ahead_distance(waypoints, closest_index, goal_index, preferred_distance, ego_state)

            #print("Min idx is: ", min_idx)
            # If there is an intersection with a stop line, update
            # the goal state to stop before the goal line.
            if min_idx is not None:
                goal_index = min_idx
                if self._traffic_light.has_changed:
                    self._traffic_light.has_changed = False

                if self._traffic_light._changed_color:
                    self._traffic_light._changed_color = False

                return goal_index, True

        return goal_index, False


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
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
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
            
            # In this case, the car is too far away.   
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return
            print("-----Lead Vehicle Distance:  ", lead_car_distance,"-------",self._follow_lead_vehicle)
            lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), 
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            #print("Angle:",np.dot(lead_car_delta_vector, 
            #          ego_heading_vector) )

            if np.dot(lead_car_delta_vector, 
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            
            
            if lead_car_distance > self._follow_lead_vehicle_lookahead + 5:
                self._follow_lead_vehicle = False
                return
            
            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 6:
                return
            
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            #print("Angle:",np.dot(lead_car_delta_vector, 
            #          ego_heading_vector) )
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False
        
    def check_for_closest_pedestrian(self,ego_state,ego_orientation,pedestrian_position,pedestrian_speed,pedestrian_rot):
        local_pos_closest=np.inf
        closest_ped_idx=None
        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x)
        if ego_angle < 0:
            ego_angle+=2*math.pi
        
        for i in range(len( pedestrian_position )):
            pedestrian_angle = math.atan2(pedestrian_rot[i][1],pedestrian_rot[i][0])
            if pedestrian_angle < 0:
                pedestrian_angle+=2*math.pi
            local_pos=from_global_to_local_frame(ego_state,pedestrian_position[i])
            if local_pos[0]>0 and local_pos[0] <30 and local_pos[1]>-5 and local_pos[1]<5 and pedestrian_angle < ego_angle-0.20 and pedestrian_angle < ego_angle-0.20:
                if local_pos[0] < local_pos_closest:
                    local_pos_closest=local_pos[0]
                    closest_ped_idx=i
                    
        if local_pos_closest == np.inf:
            self._closest_pedestrian= None
        else:
            if self._closest_pedestrian is None:
                self._closest_pedestrian={}
                self._closest_pedestrian["pos"]=pedestrian_position[closest_ped_idx]
                self._closest_pedestrian["index"]=closest_ped_idx
                self._closest_pedestrian["speed"]=pedestrian_speed[closest_ped_idx]
                self._closest_pedestrian["count"]=0
            elif self._closest_pedestrian["index"]==closest_ped_idx:
                self._closest_pedestrian["count"]+=1
                self._closest_pedestrian["pos"]=pedestrian_position[closest_ped_idx]
                self._closest_pedestrian["speed"]=pedestrian_speed[closest_ped_idx]
            else:
                self._closest_pedestrian["pos"]=pedestrian_position[closest_ped_idx]
                self._closest_pedestrian["index"]=closest_ped_idx
                self._closest_pedestrian["speed"]=pedestrian_speed[closest_ped_idx]
                self._closest_pedestrian["count"]=0
        print(self._closest_pedestrian)
        return
    def check_for_vehicle(self,ego_state, vehicle_position,vehicle_bb):
        prob_coll_vehicle=[]
        for i in range(len( vehicle_position )):
            obs_local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            if obs_local_pos[0]>0 and obs_local_pos[0] < 20 and obs_local_pos[1]<5 and obs_local_pos[1]>-5:
                prob_coll_vehicle.append(vehicle_bb[i])
        return prob_coll_vehicle
    
    def check_forward_closest_vehicle(self, ego_state, ego_orientation, vehicle_position, vehicle_rot):
        
        lead_car_idx=None
        lead_car_local_pos=None
        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x)
        if ego_angle < 0:
            ego_angle+=2*math.pi
        

        
        for i in range(len(vehicle_position)):
            vehicle_angle = math.atan2(vehicle_rot[i][1],vehicle_rot[i][0])
            if vehicle_angle < 0:
                vehicle_angle+=2*math.pi
            local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            
              
            if local_pos[0] >= 0: 
                if (vehicle_angle < ego_angle + math.pi/4 and  vehicle_angle > ego_angle - math.pi/4):  
                    if (lead_car_idx is None or local_pos[0]<lead_car_local_pos[0]) and local_pos[1]>-5 and local_pos[1]<5 :
                        lead_car_idx=i
                        lead_car_local_pos=local_pos
            
        return lead_car_idx
    
    def try_to_stop(self,waypoints,closest_index,goal_index,ego_state):
        if self._closest_pedestrian is None:
            return goal_index, False
        
        
        closest_pedestrian_local = from_global_to_local_frame(ego_state, self._closest_pedestrian["pos"])
        
        return waypoint_add_ahead_distance(waypoints,closest_index,goal_index,closest_pedestrian_local[0]-8,ego_state)
        
         
        

           
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
