print("Loading Libraries...")
from pymavlink import mavutil
import time
import numpy as np
import math
import argparse
import torch
import cv2
import time
import numpy as np
from utilsburraq import *
from yolov5 import *
import Jetson.GPIO as GPIO
print("Libraries Loaded Successfully...")
labels={
    'drowning':0,
    'swimming':1,
    'out of water':2
}
def parse_2d_list(args):
    return [args[i:i + 2] for i in range(0, len(args), 2)]
ap = argparse.ArgumentParser()
ap.add_argument("-alt", "--altitude", required=True,
   help="Altitude of the drone")
ap.add_argument("-alpha", "--alpha", required=True,
   help="Horizontal FOV")
ap.add_argument("-beta","--beta",required=True,help="Vertical FOV")
ap.add_argument("-coord_list","--coord_list", required=True,type=float, nargs='+', help='List of GPS coordinates for reaching the searching area')
ap.add_argument("-sa_list","--sa_list", required=True, type=float, nargs='+', help='List of 4 GPS coordinates of the Searching Area')
args = vars(ap.parse_args())

def main(altitude,alpha,beta,coord_list,sa_list):
    YOLO=YoloDetector("yolov5/best (8).pt")
    JetsonGPIO()
    a=input("Please press enter to continue")
    CAP=cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    OUT=cv2.VideoWriter(f"recording_{time.time()}.mp4",fourcc,12,(640,640))
    DRONE=initialize_drone()
    wait4start(DRONE)
    arm(DRONE)
    takeoff(DRONE,altitude)
    start_mission(DRONE,coord_list,sa_list,YOLO,CAP,OUT,altitude,alpha,beta,land=0)


if __name__ == '__main__':
    try:
        print("=====================================")
        print(f"Parsed Arguments: {int(args['altitude']),float(args['alpha']),float(args['beta'])}")
        coord_list=parse_2d_list(args['coord_list'])
        sa_list=parse_2d_list(args['sa_list'])
        print(f"The coordinates are: {coord_list}")
        print(f"The SA coordinates are: {sa_list}")
        print("=====================================")
        main(int(args['altitude']),float(args['alpha']),float(args['beta']),coord_list,sa_list)
    except KeyboardInterrupt:

        exit()
