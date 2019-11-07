from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image


##Detection part starts ########################################################################################################

# Make condition so that the program can run on the CPU when cuda is not available 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to(device)
model.eval()
# changed the Tensor = torch.cuda.FloatTensor to Tensor = torch.FloatTensor
classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

##Detection part end #########################################################################################################

#videopath = '../data/video/overpass.mp4'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(0)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame = vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
#outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()

while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    print("detections: ", detections)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        labels = detections[:, -1].cpu()
        print(unique_labels)
        print(labels)
        n_cls_preds = len(unique_labels)
        print(tracked_objects)
        for i, (x1, y1, x2, y2, obj_id, cls_pred) in enumerate(tracked_objects):
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)] 
            print("cls_pred: ", cls_pred)
            print("classes: ", classes)
            print("---: ", labels[i])
            print("---: ", labels[i].numpy())
            cls = classes[int(labels[i].numpy())]
            print("len(cls): ", len(cls)) 
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
           #####################################
        ##############################################################################################################################

            # params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 100,
                                   qualityLevel = 0.3,
                                   minDistance = 7,
                                   blockSize = 7 )

            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Create some random colors
            color = np.random.randint(0,255,(100,3))

            # Take first frame and find corners in it
            ret, old_frame = vid.read()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

##############################################################################################################################


            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)

    #####################################  

    cv2.imshow('Stream', frame)
    #outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
           # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
#outvideo.release()
