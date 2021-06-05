import traffic_light_detection_module 
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
        self.__bbox=None
        self.__class = None
        self.__img = None
        self.box = None
        #self._min_frames_ok = 3 #minimum number of frames 
        self._max_frame_ok = 1 #number of consecutive frames to detect traffic light 
        self._counter_consecutive_detection = 0
        self._mask = None
        self._crop_seg = None
        self._th_score = th_score
        #self.centerPoint=0
    
    def get_img(self):
        return self.__img

    def find_traffic_light(self, img):
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
        #print("SCORE: ", score)
        
        if score<self._th_score:
            self.__bbox=None
            return None
        #print("Score: ", score)
        w,h,_ = img.shape
        self.__class = box.get_label()
        self.__bbox = (int(box.xmin*w), int(box.ymin*h), int(box.xmax*w), int(box.ymax*h))
        self._counter_consecutive_detection+=1
        if self._counter_consecutive_detection < self._max_frame_ok:
            return None
        else:
            self._counter_consecutive_detection=0

        return self.__bbox


    def get_bbox(self):
        return self.__bbox
    

    def get_enlarged_bbox(self):
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


    def detect_circular_signal(self, palette_img=None):
        bbox = self.get_enlarged_bbox()
        if bbox is None or palette_img is None or self._mask is None:
            return None

        img = self.__img
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #mask = mask.reshape(mask.shape[0],mask.shape[1],1)
        #print("Shapes: ", crop_img.shape, self._mask.shape)
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)

        cv2.imshow("crop_img: ", crop_img)
        cv2.waitKey(10)
        cv2.imwrite("CropImg.jpg", crop_img)

        circles = circle_detection(blurred, display=True)
        return circles



    def get_point_with_segmentation(self, seg_img=None):

        bbox = self.get_enlarged_bbox()
        if bbox is None or seg_img is None:
            return None
        #no 0,1,2,3
        #road, lane-marking, traffic sign, sidewalk, fence, pole, wall, building, vegetation, vehicle, pedestrian, and other
        tl_label = 12
        crop_seg=seg_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        self._crop_seq = crop_seg
        tl_mask = crop_seg==tl_label
        tl_mask = tl_mask.astype(np.uint8)
        self._mask = tl_mask
        # calculate x,y coordinate of center

        if tl_mask.sum()==0:
            return None

        #print("TLMask sum: ", tl_mask.sum())

        #Use moment to find the center of a mass
        M=cv2.moments(tl_mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #self.centerPoint = (cX, cY)

        cX = bbox[0] + cX
        cY = bbox[1] + cY
        return (cX, cY)


    def is_red(self):
        return self.__class
    

    def draw_boxes_on_image(self, img):
        if self.__bbox is None or self.__img is None:
            return img
        label = self.is_red()
        c = (0,255,0) if label == 0 else (0,0,255)
        #img = np.copy(img)
        bbox = self.get_bbox()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c)
        return img 


    def draw_enlarged_boxes_on_image(self, img):
        if self.__bbox is None or self.__img is None:
            return img
        label = self.is_red()
        c = (0,255,0) if label == 0 else (0,0,255)
        #img = np.copy(img)
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

