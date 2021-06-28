from utils import centeroidnp, from_global_to_local_frame
import numpy as np
import cv2
from color_detection import traffic_color_detection

RED = 1
GREEN = 0

class TrafficLight:
    """This class is an abstraction of the traffic light.
        It stores position and color of traffic light.
        It maintain a variable to indicate if the traffic light is the next.
    """

    def __init__(self):
        self._pos = None #Last position detected for traffic light
        self._color = None
        self._measures = []
        self._cluster_belongs = None #Tuple
        self._is_next = False #it is used to say if the TL represented by this class is the next tl for the vehicle
        self._last_img_cropped = None
        self._last_mask_cropped = None
        self._prev_ok = False


    def belongs_to_cluster(self, cluster):
        if cluster[0] == self._cluster_belongs[0] and cluster[1] == self._cluster_belongs[1]:
            return True
        return False
    
    def get_position(self):
        return self._cluster_belongs
    
    def get_color(self):
        return self._color
    
    def is_next(self):
        return self._is_next
    
    def _get_tl_by_img(self):
        image = self._last_img_cropped
        mask = self._last_mask_cropped
        res = cv2.bitwise_and(image, image, mask=mask)
        return res


    def update(self, pos, color, cluster):
        """Update measurements about traffic light

        Args:
            pos (Tuple): The last position detected for traffic light
            color (int): Color detected for traffic light
            cluster (Tuple): Cluster which tl belongs
        """
        if self._last_img_cropped is not None:
            img = self._get_tl_by_img()
            color_cv = traffic_color_detection(img)
            #If color doesn't pass the threshold for detecting color, set prev_ok to False
            if color_cv is not None:
                color = color_cv 
                self._prev_ok = True
            else:
                color = self._color
                self._prev_ok = False
        


        if self._cluster_belongs is None:
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True
            

        has_changed = not self.belongs_to_cluster(cluster) #The measurements doesn't change the cluster
        if not has_changed:
            self._measures.append(pos)
            self._pos = centeroidnp(np.array(self._measures))
            self._color = color
            self._is_next = True

        else: #If the cluster has changed, we update that
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True
        

    def no_traffic_light_detection(self, ego_state): #It is called when there are no detections.
        if self._pos is None:
            return

        local_frame_pos = from_global_to_local_frame(ego_state, self._pos)
        #If this method is called and the TL is behind the vehicle, then this TL is not the next
        if local_frame_pos[0]<3: #I can't see TL anymore
            self._is_next=False
        

        


            




        
        


        

        
        


        
        





