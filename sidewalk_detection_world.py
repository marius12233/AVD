#!/usr/bin/env python3

from __future__ import print_function

import cv2
import numpy as np
from math import cos, sin, pi,tan
from sidewalk_detection import sidewalk_detection
from utils import from_local_to_global_frame

#   Required to import carla library
import os
import sys
sys.path.append(os.path.abspath(sys.path[0] + '/..'))

from lane_detection import lane_detection

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

# Lane detection and following class
class SidewalkFollowing:
    def __init__(self, camera_parameters):

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
        

    def detect(self, image, depth_data, ego_state, speed_limit = 5.0, image_rgb = None, show_lines = False,):
        height, width = image.shape

        road_mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    
        # Filter Road Mask
        # road_mask[image == 6] = 255
        road_mask[image == 7] = 255
        
        # road_mask[image == 8] = 255
        sub_mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    
        # Filter Road Mask
        # road_mask[image == 6] = 255
        sub_mask[image == 6] = 255
        road_mask+=sub_mask

        cv2.imshow("Resulting subtract", road_mask)
        cv2.waitKey(10)


        # Make lane detection
        lanes = sidewalk_detection(image, show_intermediate_steps=show_lines)

        # No Lanes ==> No suggestion
        if lanes is None or len(lanes) == 0:
            print("No point on sidewalk!")
            return []
        #print(lanes)

        lane_image = np.zeros((image.shape[0],image.shape[1]), np.uint8)
        global_lanes = []
        points = []
        #global_points = []

        for lane in lanes:
            x1,y1,x2,y2 = lane[0]
            cv2.line(lane_image, (x1,y1),(x2,y2), (255, 0, 0), 10)

            
            m =  ( (image.shape[1] - y2) - (image.shape[1] - y1) )/(x2-x1)
            b = (image.shape[1] - y1) - m*x1
            global_lanes.append([m, b])

            lane_image_temp = np.zeros((image.shape[0],image.shape[1]), np.uint8)
            
            print("Points: ", x1,y1,x2,y2)
            print("m: ", m, " b: ", b)

            #Compute points in global
            """
            x1,y1 = self.convert_point(x1, y1, depth_data, ego_state)
            x2,y2 = self.convert_point(x2, y2, depth_data, ego_state)

            #global_points.append([(x1,y1), (x2,y2)])

            #Compute m and b to obtain: y = mx+b
            if y1==y2:
                continue

            m = (x2-x1)/(y2-y1)
            b = x1 - m*y1
            global_lanes.append([m, b])
            points.append([x1, y1])
            points.append([x2, y2])
            """

            

        cv2.imshow("Lane Sidewalk", lane_image)
        cv2.waitKey(10)
        print("Global lanes: ", global_lanes)


        return global_lanes


    def convert_point(self, x, y, depth_data, ego_state):

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
        
        vehicle_frame_point = np.array([vehicle_frame_x,-vehicle_frame_y])
        #global_frame_point = from_local_to_global_frame(ego_state, vehicle_frame_point)

        return vehicle_frame_point#global_frame_point
    

