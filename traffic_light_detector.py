from traffic_light_detection_module.predict import *
import cv2
import numpy as np
from carla.image_converter import labels_to_array

def load_model():
    config = get_config(os.path.join("traffic_light_detection_module", "config.json"))
    model = get_model(config)
    return model

class TrafficLightDetector:


    def __init__(self, model, th_score=0.2):
        self.__model = model
        self.__bbox=None #bboxes on image
        self.__class = None
        self.__img = None
        self.box = None #Original box object
        self._mask = None
        self._crop_seg = None
        self._th_score = th_score
    
    def get_img(self):
        return self.__img

    def find_traffic_light(self, img):
        """Apply the detector to the image, store and return the result 

        Args:
            img (ndarray): Image to feed in the model to obtain detection

        Returns:
            (Tuple): bounding box of traffic light in the image
        """

        if img is None:
            return None
        boxes = self.__model.predict(img)
        self.__img = img
        if len(boxes)==0:
            self.__bbox=None
            return None #There is any traffic ligth
        
        box=boxes[0] #the most important
        self.box = box
        score = box.get_score()
        
        if score<self._th_score:
            self.__bbox=None
            return None
        w,h,_ = img.shape
        self.__class = box.get_label()
        self.__bbox = (int(box.xmin*w), int(box.ymin*h), int(box.xmax*w), int(box.ymax*h))

        return self.__bbox


    def get_bbox(self):
        return self.__bbox
    

    def get_enlarged_bbox(self):
        """Enlarge bbox to catch the traffic light even if it is not in the original bounding box

        Returns:
            [Tuple]: bbox enlarged
        """
        bbox = self.get_bbox()
        if bbox is None:
            return None
        xmin, ymin, xmax, ymax = bbox
        xmin-=20
        ymin-=10
        xmax+=20
        ymax+=10
        return (xmin, ymin, xmax, ymax)
    
    
    def get_img_cropped(self):
        img = self.__img
        bbox = self.get_enlarged_bbox()
        if bbox is None or img is None:
            return None
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return crop_img


    def get_point_with_segmentation(self, seg_img=None):
        """[summary]

        Args:
            seg_img (ndarray): The segmentation mask provided by Carla sensors. Defaults to None.

        Returns:
            (Tuple): coords of the center of mass of traffic light in segmentation mask
        """

        bbox = self.get_enlarged_bbox()
        if bbox is None or seg_img is None:
            return None
        #road, lane-marking, traffic sign, sidewalk, fence, pole, wall, building, vegetation, vehicle, pedestrian, and other
        tl_label = 12
        crop_seg=seg_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        self._crop_seq = crop_seg
        tl_mask = crop_seg==tl_label
        tl_mask = tl_mask.astype(np.uint8)
        self._mask = tl_mask
        if tl_mask.sum()==0:
            return None

        #Use moment to find the center of a mass
        M=cv2.moments(tl_mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cX = bbox[0] + cX
        cY = bbox[1] + cY
        return (cX, cY)


    def is_red(self):
        return self.__class
    
    #drawing methods
    def draw_boxes_on_image(self, img):
        if self.__bbox is None or self.__img is None:
            return img
        label = self.is_red()
        c = (0,255,0) if label == 0 else (0,0,255)
        bbox = self.get_bbox()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c)
        return img 


    def draw_enlarged_boxes_on_image(self, img):
        if self.__bbox is None or self.__img is None:
            return img
        label = self.is_red()
        c = (0,255,0) if label == 0 else (0,0,255)
        bbox = self.get_enlarged_bbox()
        x_c = bbox[0] + (bbox[2] - bbox[0])//2 #take the center point
        y_c = bbox[1] + (bbox[3] - bbox[1])//2
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c)
        cv2.circle(img, (int(x_c),int(y_c)), 1, (0,0,255), thickness=-1)
        return img        
    

    def show_traffic_light(self):
        if self.__img is None:
            return None
        img = self.draw_boxes_on_image()
        if img is None:
            return 
        cv2.imshow("img traffic light", img)
        cv2.waitKey(0)


if __name__=="__main__":
    detector = TrafficLightDetector()
    img = cv2.imread(os.path.join("traffic_light_detection_module", "test_images","test (8).png"))
    detector.find_traffic_light(img)
    detector.show_traffic_light()

