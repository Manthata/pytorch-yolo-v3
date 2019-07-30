# Copyright 2018 BIG VISION LLC ALL RIGHTS RESERVED
# 
from __future__ import print_function
import sys
import cv2
from random import randint
import video_demo
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import *
from darknet import Darknet
import  pandas as pd
import pickle


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])



if __name__ == '__main__':


    ####################################
    # detection codes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        "video": "video.avi", # Video to run detection upon
        "dataset": "pasacal", # Dataset on which the network has been trained
        "confidence": 0.5, # Object Confidence to filter predictions
        "nms_thresh": 0.4, # NMS Threshold
        "cfgfile": "cfg/yolov3.cfg", # Config file
        "weightsfile": "yolov3.weights", # Weightsfile
        "repo": 416 # Input resolution of the network.  Increase to increase accuracy.  Decrease to increase speed
    }

    confidence = float(params["confidence"])
    nms_thresh = float(params["nms_thresh"])

    num_classes = 80
    bbox_attr = 5 + num_classes

    bboxes = []
    xywh = []
    (H, W) = (None, None) 
    counter = 0
    line = [(43, 543), (550, 655)]


    model = Darknet(params["cfgfile"])
    model.load_weights(params["weightsfile"])
    model.net_info["height"] = params["repo"]
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    model.to(device)
    model.eval()

    #videofile = params["video"]
    #cap = cv2.VideoCapture(0)

    ##########################################



    #print("Default tracking algoritm is CSRT \n"
    #      "Available tracking algorithms are:\n")

    for t in trackerTypes:
        print(t)

    trackerType = "CSRT"

    # Set video to load
    videoPath = "videos/run.mp4"

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(0)

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        #print('Failed to read video')
        sys.exit(1)
        
    ## Select boxes
    #bboxes = video_demo.demo()
    colors = []
    #for i in bboxes:
    #    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))


    ## Initialize MultiTracker
    # There are two ways you can initialize multitracker
    # 1. tracker = cv2.MultiTracker("CSRT")
    # All the trackers added to this multitracker
    # will use CSRT algorithm as default
    # 2. tracker = cv2.MultiTracker()
    # No default algorithm specified

    # Initialize MultiTracker with tracking algo
    # Specify tracker type

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker 
    #for bbox in bboxes:
    #    print("tracking input box: ", bbox)
    #    multiTracker.add(createTrackerByName(trackerType), frame, bbox)


    frames = 0


    # Process video and track objects
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
          break
        bboxes = []


        ##########################
        # detection


        if frames%2==1:
            img, orig_im, dim = video_demo.prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)
            im_dim = im_dim.to(device)
            img = img.to(device)

            with torch.no_grad():
                output = model(Variable(img), torch.cuda.is_available())
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh)

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            #print("output: ", output)
            #print("output: ", output.shape)

            for i in output:
                x0 = i[1].int()
                y0 = i[2].int()
                x1 = i[3].int()
                y1 = i[4].int()
                #bbox = (x0, y0, x1, y1)
                #bboxes.append(bbox)
                #print(bbox)
                w = x1 - x0
                h = y1 - y0
                bboxes.append((x0, y0, w, h))
                #print(bboxes)

            # Create MultiTracker object
            multiTracker = cv2.MultiTracker_create()

            # Initialize MultiTracker 
            for bbox in bboxes:
                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                #print("tracking input box: ", bbox)
                multiTracker.add(createTrackerByName(trackerType), frame, bbox)

        ##########################

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        #print("update boxes: ", boxes)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            #print("rectangle point: ", p1, p2)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            if intersect(p1, p2, line[0], line[1]):
                counter += 1

        # draw counter
        cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
        counter += 1
        cv2.imshow('MultiTracker', frame)

        frames+=1
        #print("frames>>>>>", frames)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
