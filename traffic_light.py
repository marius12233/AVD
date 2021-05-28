from utils import centeroidnp, from_global_to_local_frame
import numpy as np

RED = 1
GREEN = 0

class TrafficLight:

    def __init__(self):
        self._pos = None
        self._color = None
        self._measures = []
        self._cluster_belongs = None #Tuple
        self._is_next = False
    
    def belongs_to_cluster(self, cluster):
        if cluster[0] == self._cluster_belongs[0] and cluster[1] == self._cluster_belongs[1]:
            return True
        return False
    
    def get_pos(self):
        return self._pos
    
    def get_color(self):
        return self._color
    
    def is_next(self):
        return self._is_next

    def update(self, pos, color, cluster):
        #If clusters differs change is next in
        
        if self._cluster_belongs is None:
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True

        has_changed = not self.belongs_to_cluster(cluster) #Se non appartiene allo stesso cluster di prima significa che le misurazioni sono cambiate
        if not has_changed:
            self._measures.append(pos)
            self._pos = centeroidnp(np.array(self._measures))
            self._color = color
            self._is_next = True

        else: #Se è cambiato
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True

            

    def no_traffic_light_detection(self, ego_state): #Viene chiamata quando non c'è detection di semafori: se non trovi più il semaforo e esso sta dietro il veicolo il semaforo corrente non è più il prossimo
        if self._pos is None:
            return
        local_frame_pos = from_global_to_local_frame(ego_state, self._pos)
        #Se è stato chiamato questo metodo e il local frame pos del traffic light sta dietro di me, allora questo semaforo non è più quello prossimo
        if local_frame_pos[0]<3.5: #Se il traffic light sta dietro il veicolo
            self._is_next=False


        
        





