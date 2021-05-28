
from __future__ import print_function

import cv2
import numpy as np
from math import cos, sin, pi,tan

#   Required to import carla library
import os
import sys
from traffic_light_detector import TrafficLightDetector, get_model
from utils import circle_detection

sys.path.append(os.path.abspath(sys.path[0] + '/..'))
from traffic_sign_recognition import TrafficLightRecognition
# Utils : X - Rotation
def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R

# Utils : Y - Rotation
def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R

# Utils : Z - Rotation
def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R

# Utils : Rotation - XYZ
def to_rot(r):
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx

class TrafficLightDetectorWorld(TrafficLightDetector):

    def __init__(self, camera_parameters, model):
        super().__init__(model)
        self.cam_height = camera_parameters['z']
        self.cam_x_pos = camera_parameters['x']
        self.cam_y_pos = camera_parameters['y']

        self.cam_yaw = camera_parameters['yaw'] 
        self.cam_pitch = camera_parameters['pitch'] 
        self.cam_roll = camera_parameters['roll']
        
        camera_width = camera_parameters['width']
        camera_height = camera_parameters['height']

        camera_fov = camera_parameters['fov']

        # Calculate Intrinsic Matrix
        f = camera_width /(2 * tan(camera_fov * pi / 360))
        Center_X = camera_width / 2.0
        Center_Y = camera_height / 2.0

        intrinsic_matrix = np.array([[f, 0, Center_X],
                                     [0, f, Center_Y],
                                     [0, 0, 1]])
                                      
        self.inv_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

        # Rotation matrix to align image frame to camera frame
        rotation_image_camera_frame = np.dot(rotate_z(-90 * pi /180),rotate_x(-90 * pi /180))

        image_camera_frame = np.zeros((4,4))
        image_camera_frame[:3,:3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]

        # Lambda Function for transformation of image frame in camera frame 
        self.image_to_camera_frame = lambda object_camera_frame: np.dot(image_camera_frame , object_camera_frame)
            
        
 
    
    def detect(self, image, depth_data, speed_limit = 5.0, seg_img=None, palette_img=None):
        #height, width, _ = image.shape
        vehicle_frame_list = []


        self.find_traffic_light(image)
        points = []

        #For now we are just using only 1 point
        if seg_img is None: #Take the center point of the bbox if there isn't seg map
            bbox = self.get_enlarged_bbox()
            
            if bbox is None:
                return None
            
            x = bbox[0] + (bbox[2] - bbox[0])//2 #take the center point of bbox
            y = bbox[1] + (bbox[3] - bbox[1])//2
            
        else:
            #Check if there are circles in segmentation image
            
            point = self.get_point_with_segmentation(seg_img) #make use of seg map to improve precision
            if point is None:
                return
            x,y = point
        """
        #Filter for traffic light signal
        img = self.get_img_cropped()
        cls_idx, score = TrafficLightRecognition().predict(img)
        #11 is the tl label
        print("TL Recognition pred, score: ", cls_idx, score)
        
        if cls_idx<9:
            if score>0.2:
                return None
        """

        points.append((x,y))


        for x,y in points:
            # From pixel to waypoint

            pixel = [x , y, 1]
            pixel = np.reshape(pixel, (3,1))
            

            # Projection Pixel to Image Frame
            depth = depth_data[y][x] * 1000  # Consider depth in meters    

            image_frame_vect = np.dot(self.inv_intrinsic_matrix, pixel) * depth
            
            # Create extended vector
            image_frame_vect_extended = np.zeros((4,1))
            image_frame_vect_extended[:3] = image_frame_vect 
            image_frame_vect_extended[-1] = 1
            
            # Projection Camera to Vehicle Frame
            camera_frame = self.image_to_camera_frame(image_frame_vect_extended)
            camera_frame = camera_frame[:3]
            camera_frame = np.asarray(np.reshape(camera_frame, (1,3)))

            camera_frame_extended = np.zeros((4,1))
            camera_frame_extended[:3] = camera_frame.T 
            camera_frame_extended[-1] = 1

            camera_to_vehicle_frame = np.zeros((4,4))
            camera_to_vehicle_frame[:3,:3] = to_rot([self.cam_pitch, self.cam_yaw, self.cam_roll])
            camera_to_vehicle_frame[:,-1] = [self.cam_x_pos, self.cam_y_pos, self.cam_height, 1]

            vehicle_frame = np.dot(camera_to_vehicle_frame,camera_frame_extended )
            vehicle_frame = vehicle_frame[:3]
            vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1,3)))

            # Add to vehicle frame list
            if abs(vehicle_frame[0][1]) < 0.5:
                # Avoid small correction for a smoother driving 
                vehicle_frame_y = 0
                vehicle_frame_x = 1.5
            else:
                vehicle_frame_y = vehicle_frame[0][1]
                vehicle_frame_x = vehicle_frame[0][0] - 1.5
            
            # -vehicle_frame_y for the inversion of the axis with the guide
            vehicle_frame_list.append([vehicle_frame_x,-vehicle_frame_y , speed_limit])

            # print(vehicle_frame_list)

        return vehicle_frame_list    
