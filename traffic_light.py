from utils import centeroidnp, from_global_to_local_frame
import numpy as np
import cv2
from color_detection import traffic_color_detection

RED = 1
GREEN = 0

class TrafficLight:

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
            pos (Tuple): [The last position detected for traffic light]
            color (int): [color detected for traffic light]
            cluster (Tuple): [cluster wichs tl belongs]
        """
        if self._last_img_cropped is not None:
            img = self._get_tl_by_img()
            color_cv = traffic_color_detection(img)
            #Se il colore non è riuscito a prenderlo bene da una telecamera il parametro prev ok viene messo a False
            #e posso decidere di usare l'altra telecamera
            if color_cv is not None:
                color = color_cv 
                self._prev_ok = True
            else:
                color = self._color
                self._prev_ok = False
            print("COLOR: ", "RED" if color else "GREEN" )
        


        if self._cluster_belongs is None:
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True
            

        has_changed = not self.belongs_to_cluster(cluster) #Se non appartiene allo stesso cluster di prima significa che le misure sono cambiate
        if not has_changed:
            self._measures.append(pos)
            self._pos = centeroidnp(np.array(self._measures))
            self._color = color
            self._is_next = True

        else: #Se è cambiato aggiornare il cluster a cui appartiene
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True
        

    def no_traffic_light_detection(self, ego_state): #Viene chiamata quando non c'è detection di semafori: se non trovi più il semaforo e esso sta dietro il veicolo il semaforo corrente non è più il prossimo
        if self._pos is None:
            return

        local_frame_pos = from_global_to_local_frame(ego_state, self._pos)
        #Se è stato chiamato questo metodo e il local frame pos del traffic light sta dietro di me, allora questo semaforo non è più il prossimo
        if local_frame_pos[0]<3: #Se il traffic light sta dietro il veicolo o avanti ma da non vederlo più
            self._is_next=False
        

        


            




        
        


        

        
        


        
        





