#!/usr/bin/env python3
from traffic_light import GREEN, RED, TrafficLight
from utils import from_global_to_local_frame, from_local_to_global_frame
import numpy as np
import math
from utils import from_global_to_local_frame, waypoint_adder_ahead, waypoints_adder_v2, waypoint_add_ahead_distance#, turn_to_intersection
from queue import PriorityQueue
# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
DECELERATE_TO_INTERSECTION = 3
OVERTAKING = 4
# Stop speed threshold
STOP_THRESHOLD = 0.03
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10
MAX_DIST_TO_STOP = 7
MIN_DIST_TO_STOP = 3
METER_TO_DECELERATE = 20

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._no_tl_found_counter = 0
        self._traffic_light:TrafficLight = None
        self._has_tl_changed_pos = False #Ci serve per dire che se il semaforo è quello di sempre mi risparmio di fare le operazioni
        self._desired_speed_intersection = 5
        self._next_intersection = None
        self._possible_overtaking = False
        self._speed_lead_car = 0
        self._lead_car = None
        self._on_current_lane = True #Set if I stay in the current lane or not
        self._may_overtake=False
        self._leads = PriorityQueue()
        self._opposites = []
        self._overtaking_vehicle = None
        self._closest_pedestrian=None
        self._nearest_intersection=None
    
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
    def set_nearest_intersection(self, nearest_intersection):
        self._nearest_intersection = nearest_intersection
    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
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
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations) 
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.
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

            #######################################################
            #CHECK FOR OVERTAKING

            if self._may_overtake and not self._on_current_lane: #Se posso sorpassare e sto nell'altra corsia: Sorpasso!!
                self._overtaking_vehicle = self._leads.get()
                self._state = OVERTAKING

            
            


            # Check traffic lights
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            
            #if self._traffic_light is not None:
            #    print("In Follow Lane: tl, tl_found", self._traffic_light.get_pos(), self._traffic_light.get_color(), self._traffic_light.is_next(), self._traffic_light.has_changed,traffic_light_found)
            
            
            
            #Se sto nell'incrocio, il semaforo era rosso ma non sono riuscito a fermarmi e
            # ormai non è + il prossimo perché le telecamere non lo vedono, passo solo se
            #non ci sono macchine contro
            #Vedere se sto alla prossima intersezione
            _,is_at_intersection =  self.check_for_next_intersection(waypoints, closest_index, goal_index, ego_state)

            if self._traffic_light is not None:
                print("Intersection: {} traffic light: {} is next: {}".format(is_at_intersection, self._traffic_light.get_color()==RED, self._traffic_light.is_next()))
                if is_at_intersection and self._traffic_light.get_color()==RED and not self._traffic_light.is_next():
                    #Controlla se c'è un veicolo in un certo range, se ci sta metti un waypoint
                    #a un metro da me per farmi fermare
                    if len(self._opposites) != 0:
                        #Fermati al closest index
                        self._goal_index = closest_index
                        self._goal_state = waypoints[closest_index]
                        self._goal_state[2] = 0
                        print("VEICOLO OPPOSTO AVANTI!!!!")
                        print("TL Waypoint: ", from_global_to_local_frame(ego_state, self._goal_state[:2]))
                        self._state = DECELERATE_TO_STOP                   
            
            try_to_stop_distance=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)
            
            
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is  None:
                try_to_stop_distance=np.inf
            
            dist=min([try_to_stop_distance,traffic_light_found_distance])
            if dist != np.inf:
                goal_index=waypoint_add_ahead_distance(waypoints,closest_index,goal_index,dist,ego_state)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                self._state = DECELERATE_TO_STOP

            

            
            

        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            color = RED
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            
            try_to_stop_distance=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is None:
                try_to_stop_distance=np.inf
            dist=min([try_to_stop_distance,traffic_light_found_distance])
            if dist != np.inf:
                goal_index=waypoint_add_ahead_distance(waypoints,closest_index,goal_index,dist,ego_state)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0

            elif  self._traffic_light is not None:
                color = self._traffic_light.get_color()
                if  (color == GREEN and self._traffic_light.is_next() ) or (color == RED and not self._traffic_light.is_next() and len(self._opposites)==0):
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
                if  (color==GREEN and self._traffic_light.is_next()) or (color == RED and not self._traffic_light.is_next() and len(self._opposites)==0):
                    self._state = FOLLOW_LANE
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            try_to_stop_distance=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)
            if try_to_stop_distance is not None:
                self._state=STAY_STOPPED
            elif tf_light:
                self._state=FOLLOW_LANE
       
        elif self._state == OVERTAKING:
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
            #  
            # Sto in questo stato fino a quando la lead_car non si trova dietro di me di tot. metri  
            #Se quello che sta al lead è quello che sto sorpassando, controllare che la distanza da questo diventi almeno -6 metri
            #Sto ancora sorpassando, tutto apposto
            if not self._overtaking_vehicle[1] == self._lead_car:
                #Non è più la lead car (si presume sia davanti a me adesso)
                #Controllare che la distanza diventi almeno 6 metri
                print("Overtaking vehicle: ", self._overtaking_vehicle)
                print("Lead car: ", self._lead_car)
                ego_dist = self._overtaking_vehicle[0]
                print("Distanza da me: ", ego_dist)
                if ego_dist < -8 :
                    self._overtaking_vehicle = None #L'ho sorpassato
                    self._state = FOLLOW_LANE
            
            try_to_stop_distance=self.try_to_stop(waypoints,closest_index,goal_index,ego_state)
            
            if try_to_stop_distance is not None:
                goal_index=waypoint_add_ahead_distance(waypoints,closest_index,goal_index,try_to_stop_distance,ego_state)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                self._state=DECELERATE_TO_STOP


                

        else:
            raise ValueError('Invalid state value.')


    def check_for_next_intersection(self, waypoints, closest_index, goal_index, ego_state,):
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
        
        #if turn_to_intersection(waypoints, self._next_intersection, ego_state):
        #    print("Girando in curva")
        #else:
        #    print("Not turn to next intersection")
        #    return goal_index, False
        #mi fermo solo se la current speed è al di sopra di un certo limite

        
        next_intersection_local = from_global_to_local_frame(ego_state, self._next_intersection[:2])
        #Prendere il waypoint con distanza minore o uguale della prossima intersezione -20
        #Controllare se il veicolo sta più lontano di 20m
        
        #Se è nell'intorno di 20m dall'intersezione diminuissci
        #il lookahead per vedere se fa meglio le curve
        if abs(next_intersection_local[0]) > METER_TO_DECELERATE :
            return goal_index, False
        return goal_index, True
        


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
            return None
        
        #Se il semaforo corrente non è più il prossimo dici che non c'è
        if not self._traffic_light.is_next():
            return None

        #Col verde non facciamo niente
        color = self._traffic_light.get_color()
        if color == GREEN:
            print("SEMAFORO VERDE!")
            return None
        

        s = np.array(self._traffic_light.get_pos()[0:2])
        s_local = from_global_to_local_frame(ego_state, s)
        
        if s_local[0] > self._lookahead: #Non sono arrivato col veicolo a guardare i waypoints nel range specificato
            return None

        
        if self._traffic_light.has_changed or self._traffic_light._changed_color:
            #SCelgo di fermarmi a distanza 6 dal semaforo
            #distanza dal semaforo
            print("CHECK FOR TRAFFIC LIGHT")

            preferred_distance = s_local[0]-MIN_DIST_TO_STOP-1
            #min_idx ,useless= waypoint_add_ahead_distance(waypoints, closest_index, goal_index, preferred_distance, ego_state)

            #print("Min idx is: ", min_idx)
            # If there is an intersection with a stop line, update
            # the goal state to stop before the goal line.
            
            
            if self._traffic_light.has_changed:
                self._traffic_light.has_changed = False

            if self._traffic_light._changed_color:
                self._traffic_light._changed_color = False

            return preferred_distance

        return None

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
            if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=15:
                self._follow_lead_vehicle_lookahead=6
            # In this case, the car is too far away.   
            if lead_car_distance >  self._follow_lead_vehicle_lookahead:
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
            
            if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=15:
                self._follow_lead_vehicle_lookahead=6
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
        if ego_angle<0:
            ego_angle+=2*math.pi
        lookahead_dist=self._lookahead
        
        for i in range(len( pedestrian_position )):
            pedestrian_angle = math.atan2(pedestrian_rot[i][1],pedestrian_rot[i][0])
            if pedestrian_angle<0:
                pedestrian_angle+=2*math.pi
            
            local_pos=from_global_to_local_frame(ego_state,pedestrian_position[i])
            if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=15:
                print("Sto all'intersezione se m appizz")
                lookahead_dist=6
            
            if local_pos[0]>0 and local_pos[0] <lookahead_dist and local_pos[1]>-5 and local_pos[1]<5 and pedestrian_angle < ego_angle-0.20 and pedestrian_angle < ego_angle-0.20:
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

            if self._overtaking_vehicle is not None and i==self._overtaking_vehicle[1]:
                prob_coll_vehicle.append(vehicle_bb[i])


        return prob_coll_vehicle
    
    def check_forward_closest_vehicle(self, ego_state, ego_orientation, vehicle_position, vehicle_rot):
        
        lead_car_idx=None
        lead_car_local_pos=None
        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x) + math.pi
        #if ego_angle < 0:
        #    ego_angle+=2*math.pi

        
        for i in range(len(vehicle_position)):
            vehicle_angle = math.atan2(vehicle_rot[i][1],vehicle_rot[i][0]) + math.pi
            #if vehicle_angle < 0:
            #    vehicle_angle+=2*math.pi
            local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            
              
            if local_pos[0] >= 0: 
                if (vehicle_angle < ego_angle + math.pi/4 and  vehicle_angle > ego_angle - math.pi/4):  
                    if (lead_car_idx is None or local_pos[0]<lead_car_local_pos[0]) and local_pos[1]>-5 and local_pos[1]<5 :
                        lead_car_idx=i
                        lead_car_local_pos=local_pos
            
        return lead_car_idx

    def check_overtaking_condition(self, ego_state, ego_orientation, vehicle_positions, vehicle_rot, vehicle_speed):
        """ Tale metodo controlla se ci sono le condizioni per sorpassare un veicolo:
            1- Non ci siano macchine di senso opposto nel giro di tot. metri
            2- Non ci sono incroci nel giro di tot. metri
            3- Non ci sono persone sulla strada nel giro di tot. metri

        Args:
            lead_car ([type]): [description]
            vehicles_position ([type]): [description]
        """

        meters = 50





        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x) + math.pi
        #if ego_angle < 0:
        #    ego_angle+=2*math.pi
        
        #Se ci sono solo lead cars allora potrei sorpassare
        lead_cars = []
        opposite_cars = []
        #1. Controllare se ci sono veicolo in senso opposto nel giro di meters
        for i in range(len(vehicle_positions)):
            lead_car = None
            opposite_car = None

            


            #Controllare se il veicolo sta di faccia
            vehicle_angle = math.atan2(vehicle_rot[i][1],vehicle_rot[i][0]) + math.pi
            #if vehicle_angle < 0:
            #    vehicle_angle+=2*math.pi

            local_pos=from_global_to_local_frame(ego_state,vehicle_positions[i][:2])

            #Aggiorniamo sempre l'overtaking vehicle
            if self._overtaking_vehicle is not None and i ==self._overtaking_vehicle[1]:
                self._overtaking_vehicle = (local_pos[0], i)
            
            
            
            if local_pos[0] >= -6 and abs(local_pos[1])<5: #Se il veicolo considerato sta davanti a me
                if (vehicle_angle < ego_angle + math.pi/4 and  vehicle_angle > ego_angle - math.pi/4):  
                    #E' un lead_car
                    lead_car = i
                else:
                    opposite_car = i

            
            #Se il veicolo ha orientamento opposto al mio Controllare sta davanti a me di tot metri
            if opposite_car:
                if local_pos[0] > meters or abs(local_pos[1])>20: #Se non sta davanti a me di tot metri e in un range di -5,+5 metri
                    opposite_car=None
            if lead_car:
                if local_pos[0] > meters or abs(local_pos[1])>20:
                    lead_car=None
            
            #Metto tutto in una PQ in base alla distanza dal veicolo
            if opposite_car:
                 #In leads ci metto in ordine in base alla distanza da me
                
                opposite_cars.append(opposite_car)
            elif lead_car:
                print("Local positions of {} is {} and v: {}".format(i, str(int(local_pos[0]))+" ,"+str(int(local_pos[1])), vehicle_speed[i-1]))
                lead_cars.append(lead_car)
                self._leads.put((local_pos[0], lead_car))
                
        
        
        #1. Prendere il prossimo lead car (il più vicino a me)
        lead_has_decelerated = False
        d=np.Inf
        min_lc = None
        for lc in lead_cars:
            distance = np.linalg.norm(np.array(vehicle_positions[lc][:2]) - np.array(ego_state[:2]))
            if distance < d:
                d = distance
                min_lc = lc

        #2. prendere la velocità
        print("Min dist lead is: ", min_lc)
        if min_lc is not None:
            speed_lead_car = vehicle_speed[min_lc]
            print("previous speed lead is: ", self._speed_lead_car)
            print("its speed is is: ", speed_lead_car)
            
            #se è zero o sta decrescendo non sorpassarlo, perché probabilmente si è fermato a un semaforo o a un incrocio
            dy = 0.5
            if speed_lead_car < 0.5 or ((speed_lead_car - self._speed_lead_car)<0 and abs(speed_lead_car - self._speed_lead_car)>dy):
                lead_has_decelerated = True

            self._speed_lead_car = speed_lead_car
            self._lead_car = min_lc
        else:
            self._speed_lead_car=0
            lead_has_decelerated = False
            self._lead_car = None
        
        self._opposites = opposite_cars
        print("Opposite cars: ", opposite_cars)
        print("leading cars: ", lead_cars)
        #Controllare se sto a 50 metri da un incrocio
        next_intersection = self._next_intersection
        if next_intersection is None:
            return False
            
        next_intersection_local = from_global_to_local_frame(ego_state, next_intersection[:2])
        if next_intersection_local[0]<meters: #Il prossimo incrocio è a meno di 50 metri
            return False

        if len(opposite_cars)==0:
            if not lead_has_decelerated:
                print("CAN OVERTAKE!!")
                print("Opposite cars: ", opposite_cars)
                print("leading cars: ", lead_cars)
                #self._follow_lead_vehicle = False
                self._may_overtake = True
                return True
            else:
                print("CANNOT OVERATE: Lead is decelerating")
                self._may_overtake = False
                return False
        else:
            print("CANNOT OVERTAKE")
            print("Opposite cars: ", opposite_cars)
            print("leading cars: ", lead_cars)
            #self._follow_lead_vehicle = False
            self._may_overtake = False
            return False
        
    
        
            

                    
    def try_to_stop(self,waypoints,closest_index,goal_index,ego_state):
        if self._closest_pedestrian is None:
            return None
        
        
        closest_pedestrian_local = from_global_to_local_frame(ego_state, self._closest_pedestrian["pos"])
        
        return closest_pedestrian_local[0]-5
        
                    




        
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
