#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models

def shutdown_hook():
    global CTRL_C, cap

    CTRL_C = True
    cap.release()
    cv.destroyAllWindows()

def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

if __name__ == "__main__":
    rospy.init_node("Feature_tracking_DeepNetwork")

    # Initialized variables
    CTRL_C = False
    rate = rospy.Rate(60)

    transformer = T.Compose([T.Resize(256),
                             T.CenterCrop(224),
                             T.ToTensor(),
                             T.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.fcn_resnet101(pretrained=True).eval().to(device)

    # Open camera
    cap = cv.VideoCapture(1)
    if not (cap.isOpened()):
        rospy.logwarn("Can't open camera 1, defaulting back to camera 0")
        cap = cv.VideoCapture(0)
        if not (cap.isOpened()):
            rospy.logfatal("Can't open camera 0. Shuting down...")
            exit()

    rospy.on_shutdown(shutdown_hook)

    while not CTRL_C:
        ret, frame = cap.read()

        im_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        im_trf = transformer(Image.fromarray(im_rgb)).unsqueeze(0)

        out = model(im_trf.to(device))["out"]
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

        rgb = decode_segmap(om)

        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

        cv.imshow('frame', bgr)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

        rate.sleep()
