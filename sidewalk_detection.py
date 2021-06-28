import cv2
import numpy as np
import math

def check_limits(value, min_value, max_value):
    value = max(value, min_value)
    value = min(value, max_value)
    return value

def generate_coords(image, line_params):
    slope, intercept = line_params
    y1 = 3*image.shape[0]/4
    y2 = image.shape[0]/2
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    x1 = check_limits(x1, 0, image.shape[1])
    x2 = check_limits(x2, 0, image.shape[1])
    
    return np.array([x1,y1,x2,y2],dtype=np.int32)

def image_sidewalk_filter_and_crop(image, use_roi = False):
    image = np.array(image, np.uint8)

    gray = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    
    # Filter Road only 7
    #We use even the central line to fullfill the road mask and detect only sidewalk lines, not the central.
    gray[image == 7] = 255
    gray[image == 6] = 255
    
    gray = cv2.dilate(gray, (5,5))
    gray = cv2.Canny(gray,150,200)

    return gray



def sidewalk_detection(image, line_optimized = True, show_intermediate_steps = True):
    height,width  = image.shape

    # Filter input image and edge detection
    canny_roi_image = image_sidewalk_filter_and_crop(image)
    

    # Lines detection
    lines = cv2.HoughLinesP(canny_roi_image, 5,  np.pi / 180, 200 * width // 800 , minLineLength=100 * width // 800, maxLineGap=30 * width // 800) # 2, np.pi / 180, 100,np.array([]), minLineLength=40, maxLineGap=5)

    return lines