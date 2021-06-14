#!/usr/bin/env python3
from traffic_light import GREEN, RED, TrafficLight
from utils import from_global_to_local_frame, from_local_to_global_frame
import numpy as np
import math
from utils import obstacle_to_world,from_global_to_local_frame, waypoint_add_ahead_distance,waypoint_precise_adder
from queue import PriorityQueue
# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
#STAY_STOPPED_PEDESTRIAN = 2
#STAY_STOPPED_TL = 3
STAY_STOPPED = 2
EMERGENCY_STOP = 4
OVERTAKING = 5
# Stop speed threshold
STOP_THRESHOLD = 0.03
EMERGENCY_STOP_THRESHOLD = 0.01
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10
MAX_DIST_TO_STOP = 7
MIN_DIST_TO_STOP = 4
METER_TO_DECELERATE = 20
DIST_FROM_PEDESTRIAN = 4

STOP_FOR_PEDESTRIAN = 0
STOP_FOR_TL = 1
#MAX_DIST_TO_STOP = 6

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
        self._forward_pedestrian = {}
        self._intersections_turn = None
        self._stop_for = None #None if you will not stop for tl or ped, 0 for pedestrian, 1 for TL
        self._pedestrian_stopped_index = None
    
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
    def set_intersections_turn(self, intersections_turn):
        self._intersections_turn = intersections_turn

    # Handles state transitions and computes the goal state.
    def transition_state_old(self, waypoints, ego_state, closed_loop_speed):
        
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
                #print("Intersection: {} traffic light: {} is next: {}".format(is_at_intersection, self._traffic_light.get_color()==RED, self._traffic_light.is_next()))
                if is_at_intersection and self._traffic_light.get_color()==RED and not self._traffic_light.is_next():
                    #Controlla se c'è un veicolo in un certo range, se ci sta metti un waypoint
                    #a un metro da me per farmi fermare
                    if len(self._opposites) != 0:
                        #Fermati al closest index
                        self._goal_index = closest_index
                        self._goal_state = waypoints[closest_index]
                        self._goal_state[2] = 0
                        #print("VEICOLO OPPOSTO AVANTI!!!!")
                        #print("TL Waypoint: ", from_global_to_local_frame(ego_state, self._goal_state[:2]))
                        self._state = DECELERATE_TO_STOP                   
            
            try_to_stop_distance=self.try_to_stop(ego_state)
            
            
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is None:
                try_to_stop_distance=np.inf
            
            if try_to_stop_distance == np.inf and  traffic_light_found_distance==np.inf:
                return
            
            elif try_to_stop_distance < traffic_light_found_distance:
                
                if 1:#self._closest_pedestrian["count"]==0 :
                    
                    print("Pedone dista , " , try_to_stop_distance)
                    goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance,0.1,ego_state)
                    #self._forward_pedestrian[self._closest_pedestrian["index"]]=goal_index
                    
                elif self._forward_pedestrian.get(self._closest_pedestrian["index"]) is not None and from_global_to_local_frame(ego_state,waypoints[self._forward_pedestrian[self._closest_pedestrian["index"]]][:2])[0] <=0:
                    print("Troppo vicino")
                    print(ego_state)
                    goal_index=waypoint_add_ahead_distance(waypoints,closest_index,goal_index,try_to_stop_distance,ego_state)
                    self._forward_pedestrian[self._closest_pedestrian["index"]]=goal_index
                else :
                    print("Uso il vecchio")
                    goal_index=self._forward_pedestrian[self._closest_pedestrian["index"]]

            else:
                #Aggiungo il waypoint al semaforo
                print("Aggiungo il waypoint al semaforo")
                goal_index=waypoint_precise_adder(waypoints,traffic_light_found_distance,0.1,ego_state)    
                self._traffic_light._changed_color = False
                self._traffic_light.has_changed = False

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
            
            try_to_stop_distance=self.try_to_stop(ego_state)
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is None:
                try_to_stop_distance=np.inf
            
            
            if try_to_stop_distance < traffic_light_found_distance:
                
                if 1:#self._closest_pedestrian["count"]==0 :
                    
                    print("Pedone dista , " , try_to_stop_distance)
                    goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance,0.1,ego_state)
                    #self._forward_pedestrian[self._closest_pedestrian["index"]]=goal_index
                    
                elif self._forward_pedestrian.get(self._closest_pedestrian["index"]) is not None and  from_global_to_local_frame(ego_state,waypoints[self._forward_pedestrian[self._closest_pedestrian["index"]]][:2])[0] <=0:
                    print("----Prima della mia pos---")
                    
                    goal_index=waypoint_add_ahead_distance(waypoints,closest_index,goal_index,try_to_stop_distance,ego_state)
                    self._forward_pedestrian[self._closest_pedestrian["index"]]=goal_index
                else:
                    print("Uso il vecchio")
                    goal_index=self._forward_pedestrian[self._closest_pedestrian["index"]]
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                self._state = DECELERATE_TO_STOP
            
            elif try_to_stop_distance >= traffic_light_found_distance and traffic_light_found_distance!=np.inf:
                goal_index=waypoint_precise_adder(waypoints,traffic_light_found_distance,0.1,ego_state)    
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                self._state = DECELERATE_TO_STOP
            
            if  try_to_stop_distance==np.Inf and len(self._opposites)==0 and self._traffic_light.get_color()==RED and not self._traffic_light.is_next():
                    self._state = FOLLOW_LANE
            
            

            

            if abs(closed_loop_speed) <= STOP_THRESHOLD and self._state != FOLLOW_LANE:
                self._state = STAY_STOPPED

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:
            tf_light=True
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            if traffic_light_found_distance is not None :
                tf_light=False
            
            try_to_stop_distance=self.try_to_stop(ego_state)
            if try_to_stop_distance is not None:
                print("Try to stop distance is not None")
                self._state=STAY_STOPPED
            elif tf_light: #and len(self._opposites)==0:
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
            if not self._overtaking_vehicle[1] == self._lead_car or self._on_current_lane:
                #Non è più la lead car (si presume sia davanti a me adesso)
                #Controllare che la distanza diventi almeno 6 metri
                #print("Overtaking vehicle: ", self._overtaking_vehicle)
                #print("Lead car: ", self._lead_car)
                ego_dist = self._overtaking_vehicle[0]
                #print("Distanza da me: ", ego_dist)
                if ego_dist < -8 or ego_dist > 3: #sta avanti a me o l'ho sorpassato
                    self._overtaking_vehicle = None #L'ho sorpassato
                    self._state = FOLLOW_LANE
            
            try_to_stop_distance=self.try_to_stop(ego_state)
            
            if try_to_stop_distance is not None:
                
                if 1:#self._closest_pedestrian["count"]==0 :
                    
                    print("Pedone dista , " , try_to_stop_distance)
                    goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance,0.1,ego_state)
                    #self._forward_pedestrian[self._closest_pedestrian["index"]]=goal_index
                    
                elif self._forward_pedestrian.get(self._closest_pedestrian["index"]) is not None and from_global_to_local_frame(ego_state,waypoints[self._forward_pedestrian[self._closest_pedestrian["index"]]][:2])[0] <=0:
                    print("----Troppo vicino---")
                    goal_index=waypoint_add_ahead_distance(waypoints,closest_index,goal_index,try_to_stop_distance,ego_state)
                    self._forward_pedestrian[self._closest_pedestrian["index"]]=goal_index
                else:
                    print("----Uso il vecchio---")
                    goal_index=self._forward_pedestrian[self._closest_pedestrian["index"]]

                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                self._goal_state[2] = 0
                self._state = DECELERATE_TO_STOP
                

        else:
            raise ValueError('Invalid state value.')

    def transition_state_middle(self, waypoints, ego_state, closed_loop_speed):
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

            # Check traffic lights
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)

            try_to_stop_distance=self.try_to_stop(ego_state)
            
            
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is None:
                try_to_stop_distance=np.inf
            
            if try_to_stop_distance == np.inf and  traffic_light_found_distance==np.inf:
                return
            
            elif try_to_stop_distance < traffic_light_found_distance:
                print("Aggiorno il waypoint al pedone a una distanza di: ", try_to_stop_distance)              
                #print("Pedone dista , " , try_to_stop_distance)
                goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance, closest_index, goal_index, 0.1, ego_state)

            else:
                #Aggiungo il waypoint al semaforo
                print("Aggiungo il waypoint al semaforo")
                goal_index=waypoint_precise_adder(waypoints,traffic_light_found_distance,closest_index, goal_index,0.1,ego_state)    

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            self._goal_state[2] = 0
            self._state = DECELERATE_TO_STOP


        elif self._state == DECELERATE_TO_STOP:
            color = RED
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self._goal_index
            
            try_to_stop_distance=self.try_to_stop(ego_state)
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is None:
                try_to_stop_distance=np.inf
            
            stop_tl = False
            if try_to_stop_distance == np.inf and  traffic_light_found_distance==np.inf:
                self._state = FOLLOW_LANE
                return     

            elif try_to_stop_distance < traffic_light_found_distance:
                print("DTS: Aggiorno il waypoint al pedone")              
                #print("Pedone dista , " , try_to_stop_distance)
                goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance,closest_index, goal_index,0.1,ego_state)

            else:
                #Aggiungo il waypoint al semaforo
                print("DTS: Aggiungo il waypoint al semaforo")
                stop_tl = True
                goal_index=waypoint_precise_adder(waypoints,traffic_light_found_distance,closest_index, goal_index,0.1,ego_state)    

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            self._goal_state[2] = 0
            self._state = DECELERATE_TO_STOP

            if abs(closed_loop_speed) <= STOP_THRESHOLD and self._state != FOLLOW_LANE:
                if stop_tl:
                    self._state = STAY_STOPPED_TL
                else:
                    self._state = STAY_STOPPED_PEDESTRIAN
        
        elif self._state == STAY_STOPPED_PEDESTRIAN:
            try_to_stop_distance=self.try_to_stop(ego_state)
            if try_to_stop_distance is None:
                self._state = FOLLOW_LANE

        elif self._state == STAY_STOPPED_TL:
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self._goal_index
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state)
            if traffic_light_found_distance is None:
                self._state = FOLLOW_LANE

    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        print("STATE: ", self._state)
        print("STOP FOR: ", self._stop_for)

        if self._state == FOLLOW_LANE:
            
            #Proviamo a diminuire il lookahead nelle curve per non far allargare troppo l'auto
            
            if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=20:
                is_turn = self._intersections_turn.get(str(self._nearest_intersection[:2]))
                if is_turn:
                    self._lookahead=15
            

            #print("FOLLOW_LANE")
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
            traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state, use_lookahead=True)

            #Check for pedestrian
            try_to_stop_distance=self.try_to_stop(ego_state)
            
            
            if traffic_light_found_distance is  None:
                traffic_light_found_distance=np.inf
            if try_to_stop_distance is None:
                try_to_stop_distance=np.inf
            
            if try_to_stop_distance == np.inf and traffic_light_found_distance==np.inf: #Non mi sto fermando
                self._pedestrian_stopped_index=None
                self._stop_for = None
                return
            
            d_real = closed_loop_speed**2/5

            if try_to_stop_distance < traffic_light_found_distance: #Mi voglio fermare per il pedone
                print("Aggiorno il waypoint al pedone a una distanza di: ", try_to_stop_distance)              
                #print("Pedone dista , " , try_to_stop_distance)
                goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance, closest_index, goal_index, 0.1, ego_state, offset=0)
                self._stop_for = STOP_FOR_PEDESTRIAN
                self._pedestrian_stopped_index = self._closest_pedestrian["index"]
                

            else: #Mi sto fermando per il semaforo
                #Aggiungo il waypoint al semaforo
                #Tuttavia se mi fermo al semaforo e sto già praticamente fermo devo andare un pò più avanti
                #Per sapere dove mi fermerò con la velocità attuale absta fare dist_current_stop = -v^2/2*a,
                #dove a = -2.5 al massimo.
                #Se dist_current_stop < dist_preferred_stop: return
                if traffic_light_found_distance > MAX_DIST_TO_STOP:
                    print("Distanza a cui mi voglio fermare: ", traffic_light_found_distance)
                    print("Distanza a cui mi fermerò: ", d_real)
                    if d_real < traffic_light_found_distance + 0.5: 
                        print("Vado troppo lento per fermarmi dove voglio")
                        return

                print("Aggiungo il waypoint al semaforo")
                goal_index=waypoint_precise_adder(waypoints,traffic_light_found_distance,closest_index, goal_index,0.1,ego_state)    
                self._stop_for = STOP_FOR_TL

            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]
            self._goal_state[2] = 0
            self._state = DECELERATE_TO_STOP



        elif self._state == DECELERATE_TO_STOP:
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self._goal_index
            print("Goal index {} speed: {} ".format(goal_index, waypoints[goal_index][2]))
                        
            #Se mi sto fermando per il pedone, quello che può capitare è che passa un pedone prima del punto
            # per cui mi sto fermando
            #Dato che la distanza dal goal index mi dice dove mi fermerò, se il nuovo pedone dista meno del goal index,
            #Allora lo aggiorno
            if self._stop_for==STOP_FOR_PEDESTRIAN:
                print("Mi sto fermando per un pedone")

                try_to_stop_distance=self.try_to_stop(ego_state)
                goal_dist = from_global_to_local_frame(ego_state, waypoints[goal_index][:2])[0]
                print("Dist to stop: ", try_to_stop_distance)
                print("Dist goal index: ", goal_dist)


                if try_to_stop_distance is None:
                    try_to_stop_distance=np.inf

                #In questo caso o il pedone è nuovo oppure è il vecchio che si sta spostando nella mia direzione.
                #Se è un pedone nuovo asggiungo un altro waypoint, ma se non riesco a fermarmi
                # dove indico con questo nuovo waypoint vado in
                # EMERGENCY STOP; altrimenti se è il vecchio calcolo la distanza da lui.
                #Se con la decelerazione massima non riesco a fermarmi prima di dove si trova il pedone
                #vado nello stato di emergency stop!
                if try_to_stop_distance < goal_dist - 0.1: 
                    d_real = closed_loop_speed**2/5
                    #goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance, closest_index, goal_index, 0.1, ego_state, offset=0)
                    if self._pedestrian_stopped_index != self._closest_pedestrian["index"]:
                        goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance, closest_index, goal_index, 0.1, ego_state, offset=0)
                        self._pedestrian_stopped_index = self._closest_pedestrian["index"]

                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                    self._goal_state[2] = 0
                    self._state = DECELERATE_TO_STOP

                    if d_real > try_to_stop_distance:
                        print("dist real: ", d_real)
                        self._state = EMERGENCY_STOP

                elif try_to_stop_distance == np.Inf: #Se la nuova distanza dal pedone è infinita (Il pedone non c'è più)
                    self._state = FOLLOW_LANE
                    self._pedestrian_stopped_index=None
                    self._stop_for=None
            
            elif self._stop_for == STOP_FOR_TL: #Se mi ero fermato per il TL può succedere che passa un pedone prima
                #In questo caso devo fermarmi prima ancora del pedone
                #Dato che il TL rimane fermo, il goal index corrente è quello più vicino al semaforo,
                #pertanto se la nuova distanza dal pedone è minore della distanza dal goal index
                #devo mettere un waypoint alla nuova distanza
                print("Mi sto fermando per un semaforo!!")
                try_to_stop_distance=self.try_to_stop(ego_state)
                if try_to_stop_distance is None:
                    self._pedestrian_stopped_index=None
                    try_to_stop_distance=np.inf

                d_real = closed_loop_speed**2/5
                if try_to_stop_distance < from_global_to_local_frame(ego_state, waypoints[goal_index][:2])[0]:
                    print("Mi stavo fermando per il semaforo -> mi fermo per il pedone")
                    print("Mi voglio fermare a distanza: ", try_to_stop_distance)
                    
                    print("Mi fermerò a distanza: ", d_real)

                    #goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance, closest_index, goal_index, 0.1, ego_state, offset=0)
                    self._pedestrian_stopped_index = self._closest_pedestrian["index"]
                    
                    self._stop_for=STOP_FOR_PEDESTRIAN

                    goal_index=waypoint_precise_adder(waypoints,try_to_stop_distance, closest_index, goal_index, 0.1, ego_state, offset=0)

                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                    self._goal_state[2] = 0 

                    if d_real > try_to_stop_distance:
                        print("dist real: ", d_real)
                        self._state = EMERGENCY_STOP


                else: #Se non è passato un pedone prima di dove mi sto fermando controllo se devo fermarmi ancora al semaforo (vedo se è diventato verde!!)
                    traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state, use_lookahead=False)
                    if traffic_light_found_distance is  None: #Se mi stavo fermando per il tl ma poi la distanza diventa infinita (Il tl è verde o non è + il prossimo)
                        self._state = FOLLOW_LANE
                        self._stop_for=None   
                    else:
                        if d_real > traffic_light_found_distance and traffic_light_found_distance <= MIN_DIST_TO_STOP -1 : #se sto a 1 metro di min dist to stop e non mi riesco a fermare ancora in tempo allora vado in EMERGENCY STOP
                            self._state = EMERGENCY_STOP

                    

            if abs(closed_loop_speed) <= STOP_THRESHOLD and self._state != FOLLOW_LANE:
                self._state = STAY_STOPPED


        elif self._state == STAY_STOPPED:
            
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self._goal_index
            if self._stop_for == STOP_FOR_PEDESTRIAN: #Se mi ero fermato per il pedone
                try_to_stop_distance=self.try_to_stop(ego_state)
                if try_to_stop_distance is None:
                    self._stop_for=None
                    self._state = FOLLOW_LANE
                    self._pedestrian_stopped_index=None

            elif self._stop_for == STOP_FOR_TL:
                traffic_light_found_distance = self.check_for_traffic_light(waypoints, closest_index, goal_index, ego_state, use_lookahead=False)
                if traffic_light_found_distance is None:
                    self._stop_for=None
                    self._state = FOLLOW_LANE

        
        elif self._state == EMERGENCY_STOP:

            if abs(closed_loop_speed) <= EMERGENCY_STOP_THRESHOLD:
                self._state = STAY_STOPPED



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
        


    def check_for_traffic_light(self, waypoints, closest_index, goal_index, ego_state, use_lookahead=True):
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
        print("CHECK FOR TRAFFIC LIGHT")
        if self._traffic_light is None:
            print("TL is None")
            return None
        
        print("STATO SEMAFORO: is next: {}, color:{}, changed:{}, changed color:{}"
        .format(self._traffic_light.is_next(), 
        self._traffic_light._color, 
        self._traffic_light.has_changed, 
        self._traffic_light._changed_color,
        
        ))
        
        #Se il semaforo corrente non è più il prossimo dici che non c'è
        if not self._traffic_light.is_next():
            print("Is not the next")
            return None

        #Col verde non facciamo niente
        color = self._traffic_light.get_color()
        if color == GREEN:
            print("SEMAFORO VERDE! skip")
            return None
        

        s = np.array(self._traffic_light.get_pos()[0:2])
        s_local = from_global_to_local_frame(ego_state, s)
        
        if use_lookahead and s_local[0] > self._lookahead: #Non sono arrivato col veicolo a guardare i waypoints nel range specificato (usato solo quando sto in follow lane)
            print("Non sono nel lookahead")
            return None

        
       
        #SCelgo di fermarmi a distanza 6 dal semaforo
        #distanza dal semaforo
        #print("CHECK FOR TRAFFIC LIGHT")

        preferred_distance = s_local[0]-MIN_DIST_TO_STOP-1
        #min_idx ,useless= waypoint_add_ahead_distance(waypoints, closest_index, goal_index, preferred_distance, ego_state)

        #print("Min idx is: ", min_idx)
        # If there is an intersection with a stop line, update
        # the goal state to stop before the goal line.
        
        """
        if self._traffic_light.has_changed:
            self._traffic_light.has_changed = False

        if self._traffic_light._changed_color:
            self._traffic_light._changed_color = False
        """

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
    def check_for_lead_vehicle(self, ego_state, lead_car_position, idx=None):
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
            
            print("#############\nLead car: ", idx)
            print("#############")
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
        """
        if ego_angle<0:
            ego_angle+=2*math.pi
        if ego_angle>math.pi:
            offset=-math.pi
        else :
            offset=+math.pi
        """
        lookahead_dist=16
        
        for i in range(len( pedestrian_position )):
            pedestrian_angle = math.atan2(pedestrian_rot[i][1],pedestrian_rot[i][0])
            
            #if pedestrian_angle<0:
            #    pedestrian_angle+=2*math.pi
            
            local_pos=from_global_to_local_frame(ego_state,pedestrian_position[i])
            if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=15:
                
                is_turn = self._intersections_turn.get(str(self._nearest_intersection[:2]))
                if is_turn:
                    lookahead_dist=6


            #print("Lookahead: ", lookahead_dist)
            diff = abs(ego_angle - pedestrian_angle)
            if diff > math.pi:
                diff = 2*math.pi - diff
                 

            if local_pos[0]> 0 and local_pos[0] <lookahead_dist and local_pos[1]>-4 and local_pos[1]<4: #and ( ((pedestrian_angle + 0.20) % (2*math.pi) < ego_angle or pedestrian_angle > (ego_angle + 0.20) % (2*math.pi)) and(  (pedestrian_angle +0.20) % (2*math.pi)  < ego_angle+offset or pedestrian_angle > (ego_angle +offset + 0.20) % (2*math.pi) )):
                

                if diff > math.pi/4 and diff < 3.5*math.pi/4:
                    if local_pos[0] < local_pos_closest:
                        #print("---------")
                        #print("Sono la macchina e vado ",ego_angle)
                        #print("Sono il pedestrian {} e vado {}".format(i,pedestrian_angle))
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
        print("Closest pedestrian: ", self._closest_pedestrian)
        return
    
    def check_for_vehicle(self,ego_state, vehicle_position,vehicle_bb, vehicle_speed, vehicle_ori, vehicle_bbox_extend):
        prob_coll_vehicle=[]
        for i in range(len( vehicle_position )):
            obs_local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            if obs_local_pos[0]>0 and obs_local_pos[0] < 20 and obs_local_pos[1]<5 and obs_local_pos[1]>-5:
                prob_coll_vehicle.append(vehicle_bb[i])

            if self._overtaking_vehicle is not None and i==self._overtaking_vehicle[1]:
                #prob_coll_vehicle.append(vehicle_bb[i])
                #Portare il veicolo + avanti
                #Prendere la locazione
                obs_local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
                #calcolare dove si troverà con s = s0 + v*t
                x=obs_local_pos[0]+vehicle_speed[i]
                y = obs_local_pos[1]
                #Ritrasformare in globale
                global_pos = from_local_to_global_frame(ego_state, [x,y])
                #Costruire bbox
                new_bbox = obstacle_to_world(global_pos, vehicle_bbox_extend[i], vehicle_ori[i])
                prob_coll_vehicle.append(new_bbox)
            """
            if i in self._opposites:
                obs_local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
                x=obs_local_pos[0] - vehicle_speed[i]/2 
                y = obs_local_pos[1]
                #Ritrasformare in globale
                global_pos = from_local_to_global_frame(ego_state, [x,y])
                #Costruire bbox
                new_bbox = obstacle_to_world(global_pos, vehicle_bbox_extend[i], vehicle_ori[i])
                prob_coll_vehicle.append(new_bbox)
            """

        return prob_coll_vehicle
    
    def check_for_pedestrian(self,ego_state, pedestrian_position,pedestrian_bb):
        prob_coll_pedestrian=[]
        for i in range(len( pedestrian_position )):
            if self._closest_pedestrian and i == self._closest_pedestrian["index"]:
                continue
            obs_local_pos=from_global_to_local_frame(ego_state,pedestrian_position[i])
            if obs_local_pos[0]>0 and obs_local_pos[0] < 16 and obs_local_pos[1]<3 and obs_local_pos[1]>-3:
                prob_coll_pedestrian.append(pedestrian_bb[i])
        return prob_coll_pedestrian
    
    def check_forward_closest_vehicle(self, ego_state, ego_orientation, vehicle_position, vehicle_rot):
        
        lead_car_idx=None
        lead_car_local_pos=None
        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x) 
        #if ego_angle < 0:
        #    ego_angle+=2*math.pi

        
        for i in range(len(vehicle_position)):
            vehicle_angle = math.atan2(vehicle_rot[i][1],vehicle_rot[i][0]) 
            #if vehicle_angle < 0:
            #    vehicle_angle+=2*math.pi
            local_pos=from_global_to_local_frame(ego_state,vehicle_position[i])
            
            diff = abs(ego_angle - vehicle_angle)
            if diff > math.pi:
                diff = 2*math.pi - diff

            if local_pos[0] >= 0: 
                if diff <= math.pi/4:  
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
        #Per ora restituiamo sempre False
        self._may_overtake = False
        return False

        meters = 50


        ego_rot_x = ego_orientation[0]
        ego_rot_y = ego_orientation[1]
        ego_angle = math.atan2(ego_rot_y,ego_rot_x)
        print("My angle: ", ego_angle) 
        #if ego_angle < 0:
        #    ego_angle+=2*math.pi
        #ego_angle_degree = ego_angle*180/math.pi
        
        #Se ci sono solo lead cars allora potrei sorpassare
        lead_cars = []
        opposite_cars = []
        #1. Controllare se ci sono veicolo in senso opposto nel giro di meters
        for i in range(len(vehicle_positions)):
            lead_car = None
            opposite_car = None

            #Controllare se il veicolo sta di faccia
            vehicle_angle = math.atan2(vehicle_rot[i][1],vehicle_rot[i][0]) #+ math.pi
            #if vehicle_angle < 0:
            #    vehicle_angle+=2*math.pi

            local_pos=from_global_to_local_frame(ego_state,vehicle_positions[i][:2])

            #Aggiorniamo sempre l'overtaking vehicle
            if self._overtaking_vehicle is not None and i ==self._overtaking_vehicle[1]:
                self._overtaking_vehicle = (local_pos[0], i)
            
            diff = abs(ego_angle - vehicle_angle)
            if diff > math.pi:
                diff = 2*math.pi - diff
            
            if local_pos[0] >= -6 and abs(local_pos[1])<5: #Se il veicolo considerato sta davanti a me
                #if (vehicle_angle < (ego_angle + math.pi/4)%(2*math.pi) and  vehicle_angle > (ego_angle - math.pi/4)%(2*math.pi) ):  
                if diff <= math.pi/4:
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
                print("Opposite angle: ", vehicle_angle)
            elif lead_car:
                #print("Local positions of {} is {} and v: {}".format(i, str(int(local_pos[0]))+" ,"+str(int(local_pos[1])), vehicle_speed[i-1]))
                lead_cars.append(lead_car)
                print("Lead angle: ", vehicle_angle)
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
        #print("Min dist lead is: ", min_lc)
        if min_lc is not None:
            speed_lead_car = vehicle_speed[min_lc]
            #print("previous speed lead is: ", self._speed_lead_car)
            #print("its speed is is: ", speed_lead_car)
            
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
            self._may_overtake = False
            print("CANNOT OVERTAKE: next_intersection is None")
            return False
            
        next_intersection_local = from_global_to_local_frame(ego_state, next_intersection[:2])
        if next_intersection_local[0]<meters: #Il prossimo incrocio è a meno di 50 metri
            self._may_overtake = False
            print("CANNOT OVERTAKE: Il prossimo incrocio è a meno di 50 metri")
            return False
        
        #Controllare anche che non sia in un incrocio
        if self._nearest_intersection and np.linalg.norm(np.array(self._nearest_intersection[:2]) - np.array(ego_state[:2]) )<=20:
            self._may_overtake = False
            print("CANNOT OVERTAKE: Sto a 20m dall'incrocio")
            return False
        
        #Controllare anche se l'incrocio più vicino dietro di me sta a 50m

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
                       
    def try_to_stop(self,ego_state):
        if self._closest_pedestrian is None:
            return None

        closest_pedestrian_local = from_global_to_local_frame(ego_state, self._closest_pedestrian["pos"])
        return closest_pedestrian_local[0]-DIST_FROM_PEDESTRIAN
        
                    




        
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
