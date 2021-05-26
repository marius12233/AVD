import cv2
import numpy as np
from math import cos, sin, pi,tan
from scipy.spatial import distance
from utils import from_global_to_local_frame, from_local_to_global_frame, centeroidnp

class TrafficLightTracking:

    def __init__(self):
        self.groups = {}
        self.color_groups = {}
        self.directions = {}
        self.max_meters = 10
        self._max_distance_to_vehicle = 60
        self.min_measurements = 5
    
    def track(self,ego_state, vehicle_frame_pos, color):
        x_global, y_global = from_local_to_global_frame(ego_state, vehicle_frame_pos)        
        self.update(ego_state, (x_global, y_global), color)
    

    def update(self, ego_state, pos_global, color):
        x_global, y_global = pos_global
        
        d=np.inf
        min_dist_elem = None
        clusters_to_delete = []
        for k in self.groups.keys():
            #Compute the location of cluster respect to vehicle frame
            print("Group meas: ", k)
            #print("TL meas: ", (x_global, y_global))
            x_l, y_l = from_global_to_local_frame(ego_state, k)

            print("Local coords: ", x_l, y_l)

            if x_l > 0 and y_l > 0:
                print("Is at Right")

            if x_l < 0:
                print("REMOVE K: ", k)
                clusters_to_delete.append(k)
                continue


            dist = distance.euclidean(np.array(k), np.array((x_global, y_global))) #calcolo la distanza tra un cluster e una misura
            print("Meas {} cluster {} dist: {}".format(pos_global, k, dist))
            if dist<d and dist<self.max_meters:
                #print("Dist: ", np.sqrt((k[0]-x_global)**2 + (k[1]-y_global)**2) )
                d = dist
                min_dist_elem = k

        for k in clusters_to_delete:
            del(self.groups[k])
        
        if min_dist_elem is not None:
            self.groups[min_dist_elem].append([x_global, y_global])
            self.color_groups[min_dist_elem].append(color)
        
        else: #create another group
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
            #add condition that it has to be forward
            
            print("my_distance: ", dist)
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
        return (centroid, color)


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
        





                




        

