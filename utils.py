import cv2
import numpy as np
from math import cos, sin, pi,tan

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def from_local_to_global_frame(ego_state, vehicle_frame_pos):
    x,y = vehicle_frame_pos

    
    x_global = ego_state[0] + x*cos(ego_state[2]) - \
                                    y*sin(ego_state[2])
    y_global = ego_state[1] + x*sin(ego_state[2]) + \
                                    y*cos(ego_state[2])
    return x_global, y_global


def from_global_to_local_frame(ego_state, global_frame_pos):
    x_global, y_global = global_frame_pos
    x_g = x_global - ego_state[0]
    y_g = y_global - ego_state[1]
    x_l =  x_g*cos(ego_state[2]) + \
                                    y_g*sin(ego_state[2])

    y_l =  - x_g*sin(ego_state[2]) + \
                                    y_g*cos(ego_state[2])
    return x_l, y_l


def circle_detection(img, display=False):

    output = img.copy()
    minDist = 100
    param1 = 30 #500
    param2 = 50 #200 #smaller value-> more false circles
    minRadius = 5
    maxRadius = 100 #10

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # ensure at least some circles were found
    if circles is not None:
        print("CIRCLE DETECTED!!")
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        if display:
            # show the output image
            cv2.imshow("Circle", output)
            cv2.waitKey(10)
    
    return circles

    
        