import cv2
import numpy as np

lower_yellow = np.array([15, 10,10])
upper_yellow = np.array([36, 255, 255])	

lower_red = np.array([0,10,10])
upper_red = np.array([10,255,255])

lower_green = np.array([36,10,10])
upper_green = np.array([70, 255, 255])

RED = 1
GREEN = 0



def detect(image, color="red"):
    if color=="red":
        lower = lower_red
        upper = upper_red
    elif color == "yellow":
        lower = lower_yellow
        upper = upper_yellow
    elif color == "green":
        lower = lower_green
        upper = upper_green
       

    image = cv2.resize(image, (224,224))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    return int(((mask/255).sum()))#/(mask.shape[0]*mask.shape[1]) #To obtain the percentage



def traffic_color_detection(image, threshold=50):
    green_area = detect(image, color="green")
    yellow_area = detect(image, color="yellow")
    red_area = detect(image, color="red")
    areas = np.array([green_area, yellow_area, red_area])
    print("areas: ", areas)
    max_area = np.max(areas)
    if max_area<threshold:
        print("Not passed threshold: ", max_area)
        return None

    if red_area > green_area or yellow_area > green_area:
        return RED
    else:
        return GREEN

if __name__=="__main__":
    image = cv2.imread("red_tl/tl_32.jpg")
    print("Color: ", traffic_color_detection(image))

