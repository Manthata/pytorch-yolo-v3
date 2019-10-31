from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
from pyimagesearch.centroidtracker import CentroidTracker
import dlib 
import imutils
import argparse
from imutils.video import FPS
from pyimagesearch.trackableobject import TrackableObject


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    print("c1: ", c1)
    print("c2: ", c2)
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def demo():

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
    nms_thesh = float(params["nms_thresh"])
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    bbox_attrs = 5 #num_classes

    bboxes = [] 
    xywh = []

    print("Loading network.....")
    model = Darknet(params["cfgfile"])
    model.load_weights(params["weightsfile"])
    print("Network successfully loaded")

    model.net_info["height"] = params["repo"]
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    videofile = params["video"]
    # activate our centroid tracker 
    (H, W) = (None, None)
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    
    # set 0 for debug
    cap = cv2.VideoCapture(0)
    fps = FPS().start()
    rects= []
    status = "Waiting.."
    

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
                                               
            for i in output:
                x0 = i[1].int()
                y0 = i[2].int()
                x1 = i[3].int()
                y1 = i[4].int()
                bbox = (x0, y0, x1, y1)
                bboxes.append(bbox)
                print(bbox)
                w = x1 - x0
                h = y1 - y0
                xywh.append((x0, y0, w, h))
                print(x0, y0, w, h)
                                               
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x0, y0, x1, y1)
                tracker.start_track(rgb, rect)
                                               
                trackers.append(tracker)
                                               
            for tracker in trackers:
                # set the status of the system to tracking
                status = "Tracking.."
                   # update the tracker and grap the update position 
                tracker.update(rgb)
                pos = tracker.get_position()
                # Unpack the position 
                x0 = int(pos.left())
                y0 = int(pos.top())
                x1 = int(pos.right())
                y1 = int(pos.bottom())
                #add the bounding box coordiants to the rectangle
                rects.append((x0, y0, x1, y1))
                 # moving 'up' or 'down'
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 2)
            objects = ct.update(rects)  
            # Loop through the tracked objects
            for (objectID, centroid) in objects.items():                
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID,centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] -np.mean(y)
                    to.centroids.append(centroid)
                    if not to.counted :
                        # if the direction is negative 
                        # indicatin gthe object is moving up 
                        # and the centroid is above the center line 
                        # count the object 
                        if direction < 0 and centroid[1]< h // 2:
                            totalUp +=1
                            to.counted = True
                        # if the direction is positive
                        # indicating the object is moving down
                        # and centroid is below the center line
                        elif direction > 0 and centroid[1] > h // 2:
                            totalDown += 1
                            to.counted = True
                       
                    # store the trackable object in the dictionary 
                    trackableObjects[objectID] = to 
                    
                #draw both the ID of the object and the centroid of the object
                # on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), 
                           -1)
                info = [
                    ("Up", totalUp),
                    ("Down", totalDown),
                    ("Status", status)
                ]
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k,v)
                    cv2.putText(frame, text, (10, h - ((i*20) +20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 225), 2)
                    
            #return bboxes
            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            # write bbox
            list(map(lambda x: write(x, orig_im, classes, colors), output))

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            fps.update()
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
     
            #return xywh

        else:
            break


if __name__ == '__main__':
  demo()
