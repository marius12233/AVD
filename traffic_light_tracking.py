import cv2
import numpy as np
from math import cos, sin, pi,tan
from scipy.spatial import distance
from utils import from_global_to_local_frame, from_local_to_global_frame, centeroidnp
from pykalman import KalmanFilter

MAX_DIM_COLORS = 7

def apply_kalman_filter(measurements, show=False):
    if len(measurements) <2:
        return None
    initial_state_mean = [measurements[0, 0],
                        0,
                        measurements[0, 1],
                        0]

    transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]

    kf = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean, n_dim_obs=2)

    kf1 = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)
    return (smoothed_state_means[-1][0], smoothed_state_means[-1][2])

class TrafficLightTracking:

    def __init__(self, intersection_nodes=None):
        self.groups = {}
        self.color_groups = {}
        self.directions = {}
        self.max_meters = 10
        self._max_distance_to_vehicle = 60
        self.min_measurements = 5
        self._intersection_nodes = intersection_nodes

        ## KALMAN FILTER meas
        self._measurements = []
        self._kf_pos = None


    def find_next_intersection(self, ego_state):
        intersection_nodes = self._intersection_nodes
        if intersection_nodes is None:
            return None
        #Find the intersection at minimum distance which is forward
        dist = np.Inf #distance between your x and intersection x
        next_intersection = None
        for point in intersection_nodes:
            x_p, y_p = point[0], point[1]
            local_point = from_global_to_local_frame(ego_state, (x_p, y_p))
            if local_point[0]>0 and abs(local_point[1])<30: #se sta davanti al veicolo e in un range di 10 metri
                distance = np.linalg.norm(np.array(point[:2])-np.array(ego_state[:2]))
                if distance<dist: #Lo prendo a minima distanza da me
                    dist = distance
                    next_intersection = point

        return next_intersection


    def find_nearest_intersection(self, ego_state):
        intersection_nodes = self._intersection_nodes
        if intersection_nodes is None:
            return None
        #Find the intersection at minimum distance which is forward
        dist = np.Inf #distance between your x and intersection x
        next_intersection = None
        for point in intersection_nodes:
            x_p, y_p = point[0], point[1]
            #local_point = from_global_to_local_frame(ego_state, (x_p, y_p))

            #if local_point[0]>0: #se sta davanti

            distance = np.linalg.norm(np.array(ego_state[:2])-np.array(point[:2]))

            if distance<dist: #Lo prendo a minima distanza da me
                dist = distance
                next_intersection = point

        return next_intersection

    def get_kf_pos(self):
        return self._kf_pos        


    def track(self,ego_state, vehicle_frame_pos, color):
        x_global, y_global = from_local_to_global_frame(ego_state, vehicle_frame_pos)       
        self.update(ego_state, (x_global, y_global), color)
    

    def update(self, ego_state, pos_global, color):
        x_global, y_global = pos_global
        ### USE KALMAN FILTER
        self._measurements.append((x_global, y_global))
        if len(self._measurements)>self.min_measurements:
            meas = np.asarray(self._measurements)
            print("meas:", meas)
            print("Size: ", meas)
            self._kf_pos = apply_kalman_filter(meas)


        #Filter points
        #Use distance from next intersection to filter out data from other signals
        
        #Find my next intersections

        #Filtra i punti che hanno distanza maggiore di 20 dalla prossima intersezione
        pos_local = from_global_to_local_frame(ego_state, pos_global)
        next_intersection = self.find_next_intersection(ego_state)

        if next_intersection is None: #probably I stay in a curve
            return
        
        next_intersection = next_intersection[:2]
        print("Next intersection: ", next_intersection)
        next_intersection_local = from_global_to_local_frame(ego_state, next_intersection)
        dist_next_intersection_to_point = np.linalg.norm(np.array(pos_global) - np.array(next_intersection))
        if  dist_next_intersection_to_point > 20: #se il punto trovato ha distanza maggiore di 20m  o sta davanti la prossima intersezione, devo scartarlo
            print("Distance from next intersection: ", dist_next_intersection_to_point)
            print("point: {} , intersection: {}".format(pos_local, next_intersection_local))
            return
         
        

        #Calcolo l'intersezione più vicina
        nearest_intersection = self.find_nearest_intersection(ego_state)
        #La porto in terna veicolo
        nearest_intersectioin_local = from_global_to_local_frame(ego_state, nearest_intersection[:2])
        #se la x del nearest intersection è <0 significa che al momento della misurazione 
        #ho già passato l'incrocio più vicino
        pos_local = from_global_to_local_frame(ego_state, pos_global)
        #Se la misurazione che sto effettuando sta più avanti dell'incrocio più vicino, skippala
        if pos_local[0] > nearest_intersectioin_local[0]:
            return
        


        d=np.inf
        min_dist_elem = None
        clusters_to_delete = []
        for k in self.groups.keys():
            #Compute the location of cluster respect to vehicle frame
            #print("Group meas: ", k)
            #print("TL meas: ", (x_global, y_global))
            x_l, y_l = from_global_to_local_frame(ego_state, k)

            #print("Local coords: ", x_l, y_l)

            #if x_l > 0 and y_l > 0:
            #    print("Is at Right")

            if x_l < 0:
                #print("REMOVE K: ", k)
                clusters_to_delete.append(k)
                continue


            dist = distance.euclidean(np.array(k), np.array((x_global, y_global))) #calcolo la distanza tra un cluster e una misura
            #print("Meas {} cluster {} dist: {}".format(pos_global, k, dist))
            if dist<d and dist<self.max_meters:
                #print("Dist: ", np.sqrt((k[0]-x_global)**2 + (k[1]-y_global)**2) )
                d = dist
                min_dist_elem = k

        for k in clusters_to_delete:
            del(self.groups[k])
        
        if min_dist_elem is not None:
            self.groups[min_dist_elem].append([x_global, y_global])
            self.color_groups[min_dist_elem].append(color)
            if len(self.color_groups[min_dist_elem])>MAX_DIM_COLORS:
                self.color_groups[min_dist_elem] = self.color_groups[min_dist_elem][-MAX_DIM_COLORS:]
        
        else: #create another group
            x_c, y_c = int(x_global), int(y_global)
            self.groups[(x_c,y_c)] = [[x_global, y_global]]
            self.color_groups[(x_c,y_c)] = [color]


    def get_nearest_tl(self, ego_state):
        #compute distance between each center of cluster
        nearest_intersection = self.find_nearest_intersection(ego_state)#Take the nearest intersection to the cluster
        nearest_intersection = (nearest_intersection[0], nearest_intersection[1])
        
        d=np.inf
        min_dist_elem = None
        for k in self.groups.keys():
            dist = distance.euclidean(np.array(k), np.array(ego_state[:2]))
            x_l, y_l = from_global_to_local_frame(ego_state, k)
            #add condition that it has to be forward
            
            #################à FILTER IF THE INTERSECTION IS ASSOCIATED TO THE CLUSTER ######à
            #Devo considerare che se ho un elemento a distanza minima che va bene come cluster posso cambiare l'associazione con l'incrocio
            #IDEA:
            # #Oltre al cluster, nell'intersezione metto anche la distanza, relativa al veicolo
            # del cluster rispetto all'intersezione 
            # In questo if devo controllare che la distanza relativa al veicolo dal punto di int. non sia <0)
            #if self._track_intersection_tl[nearest_intersection] is not None and not self._track_intersection_tl[nearest_intersection]==k: #Se ho associato l'intersezione già a un cluster e quel cluster non è quello che sto considerando, vai avanti
            #   continue
            #############################################################


            #print("my_distance: ", dist)
            if dist<d and len(self.groups[k])>self.min_measurements and x_l > 0 and y_l>0:
                d = dist
                min_dist_elem = k
        

        if min_dist_elem is None:
            return None

        points = self.groups[min_dist_elem]
        colors = self.color_groups[min_dist_elem]


         
        
        
        #del(self.groups[min_dist_elem])
        #del(self.color_groups[min_dist_elem])

        #get cettroid
        centroid = centeroidnp(np.array(points))
        sum_colors = np.sum(colors)
        color = 1 if  sum_colors > len(colors)//2  else 0 #Majority vote
        #color = colors[-1]
        return (centroid, color), min_dist_elem


    def get_clusters(self):
        return {k:len(self.groups[k]) for k in self.groups.keys()}



        

if __name__=="__main__":
    tracker = TrafficLightTracking()
    tracker.track((170, 130, 38), (104.596,128.16), 0)
    tracker.track((139, 129.8, 38), (104.61,128.12), 0)
    tracker.track((138, 130, 38), (104.61,128.07), 0)
    tracker.track((137, 130, 38), (104.61,128.07), 0)
    tracker.track((136, 130, 38), (104.61,128.07), 0)
    tracker.track((136, 130, 38), (104.61,128.07), 0)
    tracker.track((136, 130, 38), (104.61,128.07), 0)
    tracker.track((132, 130, 38), (104.61,128.07), 0)
    tracker.track((136, 130, 38), (104.61,128.07), 0)
    tracker.track((129, 130, 38), (104.61,128.07), 0)
    tracker.track((136, 130, 38), (104.61,128.07), 0)
    tracker.track((121, 130, 38), (104.61,128.07), 0)
    tracker.track((100, 127, 38), (104.61,128.07), 0)
    tracker.track((99, 127, 38), (104.62,128.27), 0)
    print(tracker.get_nearest_tl((99, 127, 38)))

    print("D: ",distance.euclidean((99, 127), (104.62,128.27)))
        





                




        

