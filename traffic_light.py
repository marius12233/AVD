from utils import centeroidnp, from_global_to_local_frame
import numpy as np
import cv2
from color_detection import traffic_color_detection

RED = 1
GREEN = 0

class TrafficLight:

    def __init__(self):
        self._pos = None
        self._color = None
        self._measures = []
        self._cluster_belongs = None #Tuple
        self._is_next = False
        self.has_changed = True #Questo parametro verrà utilizzato da una classe esterna e verrà settato a False quando viene usato
        self._last_img_cropped = None
        self._last_mask_cropped = None

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
    
    def get_cluster_centroid(self):
        return self._cluster_belongs 
    
    def _get_tl_by_img(self):
        image = self._last_img_cropped
        mask = self._last_mask_cropped
        res = cv2.bitwise_and(image, image, mask=mask)
        return res


    def update(self, pos, color, cluster):
        if self._last_img_cropped is not None:
            img = self._get_tl_by_img()
            color_cv = traffic_color_detection(img)
            color = color_cv 
            print("COLOR: ", "RED" if color else "GREEN" )
        
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
            self.has_changed = True
        
        if self._last_img_cropped is not None and self._last_mask_cropped is not None:
            if self._color == 1:
                res = self._get_tl_by_img()
                #cv2.imwrite("data_collected/tl_"+str(np.random.randint(1000))+".jpg", res)
                cv2.imshow("TL: ", res)
                cv2.waitKey(10)

            

    def no_traffic_light_detection(self, ego_state): #Viene chiamata quando non c'è detection di semafori: se non trovi più il semaforo e esso sta dietro il veicolo il semaforo corrente non è più il prossimo
        if self._pos is None:
            return
        local_frame_pos = from_global_to_local_frame(ego_state, self._pos)
        #Se è stato chiamato questo metodo e il local frame pos del traffic light sta dietro di me, allora questo semaforo non è più quello prossimo
        if local_frame_pos[0]<0: #Se il traffic light sta dietro il veicolo
            self._is_next=False
        
        

        #Se è stata chiamata questo metodo ma il traffic light non sta dietro di me e il veicolo sta fermo, 
        #Alg:
        #Prendo l'immagine, applico la maschera per prendermi solo il semaforo
        #Converto i colori in hsv per classificare meglio
        #FARLO SOLO SE SEI IL NEXT 
        #TODO: if self._is_next:
        """
        if self._last_img_cropped is not None and self._last_mask_cropped is not None:
            if self._color == 1:
                image = self._last_img_cropped
                mask = self._last_mask_cropped
                res = cv2.bitwise_and(image, image, mask=mask)
                #cv2.imwrite("tl_"+str(np.random.randint(1000))+".jpg", res)
                cv2.imshow("TL: ", res)
                cv2.waitKey(10)
        """
        


        
        





