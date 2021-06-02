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
        self._changed_color = False
        self.i = 0
        self._bbox = None
        self._complete_img = None
        self._seg_img = None
    
    def set_complete_image(self, complete_img):
        self._complete_img = complete_img
    
    def set_bbox(self, bbox):
        self._bbox = bbox
    
    def set_seg_img(self, seg_img):
        self._seg_img = seg_img

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
            if color_cv is not None:
                color = color_cv 
            else:
                color = self._color
            print("COLOR: ", "RED" if color else "GREEN" )

        if not color == self._color:
            self._changed_color=True

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

        else: #Se è cambiato
            self._cluster_belongs = cluster
            self._measures = [pos]
            self._pos = pos
            self._color = color
            self._is_next = True
            self.has_changed = True
        
        if self._last_img_cropped is not None and self._last_mask_cropped is not None:
            #if self._color == 1:
            res = self._get_tl_by_img()
            color_cv = traffic_color_detection(img)
            color_name = "void"
            if self._color==1: 
                color_name = "red"  
            elif self._color==0:
                color_name = "green"
            """
            cv2.imwrite("data_"+color_name+"/tl_"+str(self.i)+".jpg", res)
            self.i+=1
            cv2.imshow("TL: ", res)
            cv2.waitKey(10)
            """

            

    def no_traffic_light_detection(self, ego_state, stopped): #Viene chiamata quando non c'è detection di semafori: se non trovi più il semaforo e esso sta dietro il veicolo il semaforo corrente non è più il prossimo
        if self._pos is None:
            return

        local_frame_pos = from_global_to_local_frame(ego_state, self._pos)
        #Se è stato chiamato questo metodo e il local frame pos del traffic light sta dietro di me, allora questo semaforo non è più il prossimo
        if local_frame_pos[0]<3: #Se il traffic light sta dietro il veicolo
            self._is_next=False
        

        
        if self._bbox is not None and self._complete_img is not None:
            bbox = self._bbox
            img = self._complete_img
            seg_img = self._seg_img

            
            #no 0,1,2,3
            #road, lane-marking, traffic sign, sidewalk, fence, pole, wall, building, vegetation, vehicle, pedestrian, and other
            print("no_traffic_light_detection", bbox is not None and img is not None and seg_img is not None)
            print("State: ", stopped)
            if bbox is not None and img is not None and seg_img is not None:
                tl_label = 12
                xmin = bbox[0]-10
                ymin = bbox[1] - 10
                xmax = bbox[2] +10
                ymax = bbox[2] +10
                crop_seg=seg_img[ymin:ymax, xmin:xmax]
                tl_mask = crop_seg==tl_label
                tl_mask = tl_mask.astype(np.uint8)
                crop_img = img[ymin:ymax, xmin:xmax]
                res = cv2.bitwise_and(crop_img, crop_img, mask=tl_mask)
                cv2.imshow("Mask res: ", res)
                cv2.waitKey(10)
                color_cv = traffic_color_detection(res)
                print("Color in no measure: ", color_cv)
                if color_cv and color_cv != self._color and (stopped==2 or stopped==1):
                    self._color = color_cv
                    self._changed_color = True
                    print("Color in no measure: ", color_cv)

            




        
        


        

        
        


        
        





