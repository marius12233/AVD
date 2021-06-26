import cv2
import numpy as np
from math import cos, sin, pi,tan
from scipy.spatial import distance
from utils import from_global_to_local_frame, from_local_to_global_frame, centeroidnp
from pykalman import KalmanFilter

MAX_DIM_COLORS = 5
LEN_ROAD = 10
TL_RANGE_INTERSECTION = 20


class TrafficLightTracking:

    def __init__(self, intersection_nodes=None):
        self.groups = {} #clusters to contains measurements
        self.color_groups = {} #mantains the last 7 measurements for the color
        self.max_meters = 15 #The radius of the circle to inglobe the measurements
        self.min_measurements = 3 #the minimum measurements we need to consider a cluster
        self._intersection_nodes = intersection_nodes
        

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
            if local_point[0]>0 and abs(local_point[1])<LEN_ROAD: #se sta davanti al veicolo e in un range di 10 metri lateralmente
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
            distance = np.linalg.norm(np.array(ego_state[:2])-np.array(point[:2]))
            if distance<dist: #Lo prendo a minima distanza da me
                dist = distance
                next_intersection = point

        return next_intersection


    def track(self,ego_state, vehicle_frame_pos, color):
        x_global, y_global = from_local_to_global_frame(ego_state, vehicle_frame_pos)       
        self.update(ego_state, (x_global, y_global), color)
    

    def update(self, ego_state, pos_global, color):
        x_global, y_global = pos_global
        #Filter points
        #Use distance from next intersection to filter out data from other signals
        #Find my next intersection
        pos_local = from_global_to_local_frame(ego_state, pos_global)
        next_intersection = self.find_next_intersection(ego_state)

        if next_intersection is None: #probably I stay in a curve, where there are no traffic lights
            return
        
        next_intersection = next_intersection[:2]
        #In Carla traffic lights are in a range of 20 meters from intersections.
        #We use this value to filter out points not in that range
        next_intersection_local = from_global_to_local_frame(ego_state, next_intersection)
        dist_next_intersection_to_point = np.linalg.norm(np.array(pos_global) - np.array(next_intersection))
        if  dist_next_intersection_to_point > TL_RANGE_INTERSECTION: #se il punto trovato ha distanza maggiore di 20m  o sta davanti la prossima intersezione, devo scartarlo
            return

        #If the current measurement is further than the nearest intersection or 
        #it is on the left of the vehicle, we skip that
        if pos_local[0] > next_intersection_local[0] or pos_local[1]<0:
            return

        d=np.inf
        min_dist_elem = None
        clusters_to_delete = []
        for k in self.groups.keys():
            #Compute the location of cluster respect to vehicle frame
            x_l, y_l = from_global_to_local_frame(ego_state, k)

            if x_l < 0 or abs(y_l) > 5: #if the cluster isn't around 5 meters to vehicle or it is behind the vehicle, we delete the cluster
                clusters_to_delete.append(k)
                continue

            dist = distance.euclidean(np.array(k), np.array((x_global, y_global))) #compute the distance between a cluster and the current measurrement
            if dist<d and dist<self.max_meters: #We are lookling for the closest cluster to the measurement in a range of 15meters
                d = dist
                min_dist_elem = k

        for k in clusters_to_delete:
            del(self.groups[k])
        
        if min_dist_elem is not None: #update group/cluster
            self.groups[min_dist_elem].append([x_global, y_global])
            self.color_groups[min_dist_elem].append(color)
            if len(self.color_groups[min_dist_elem])>MAX_DIM_COLORS: 
                self.color_groups[min_dist_elem] = self.color_groups[min_dist_elem][-MAX_DIM_COLORS:]
        
        else: #create another group/cluster
            x_c, y_c = int(x_global), int(y_global)
            self.groups[(x_c,y_c)] = [[x_global, y_global]]
            self.color_groups[(x_c,y_c)] = [color]


    def get_nearest_tl(self, ego_state):
        #compute distance between each center of cluster
        d=np.inf
        min_dist_elem = None
        for k in self.groups.keys():
            dist = distance.euclidean(np.array(k), np.array(ego_state[:2]))
            x_l, y_l = from_global_to_local_frame(ego_state, k)

            if dist<d and len(self.groups[k])>=self.min_measurements and x_l > 0:
                d = dist
                min_dist_elem = k
        
        if min_dist_elem is None:
            return None

        points = self.groups[min_dist_elem]
        colors = self.color_groups[min_dist_elem]

        centroid = centeroidnp(np.array(points)) #Compute the centroid of measurements
        sum_colors = np.sum(colors)
        color = 1 if  sum_colors > len(colors)//2  else 0 #Majority vote to choose color
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
        





                




        

