import argparse
import copy
import json
import math
import os
import re
from collections import deque

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image

RGB_FOLDER = ['frontcamerarecord', 'downcamerarecord']


class DinoMonitor:

    def __init__(self, device=0):
        self.dino_model = None
        self.init_dino_model(device)
        self.object_desc_dict = dict()
        self.init_object_dict()

    def init_object_dict(self):
        with open('data/config/object_new_name.json', 'r') as f:
            file = json.load(f)
            for item in file:
                self.object_desc_dict[item['object_name']] = item['new_name']

    def init_dino_model(self, device):
        import sys
        sys.path.append('GroundingDINO')
        from groundingdino.util.inference import load_model, predict
        device = torch.device(device)
        from functools import partial
        model = load_model(
            "utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "utils/GroundingDINO/groundingdino_swint_ogc.pth")
        model.to(device=device)
        self.dino_model = partial(predict, model=model)

    def get_dino_results(self, rgb, depth, obj_info):
        # here we temporarily assume that bs=1
        images = [rgb]
        depths = [depth]
        done = False

        for i in range(len(images)):
            img = images[i]
            depth = depths[i]
            if self.dino_target_detection(img, depth, obj_info):
                done = True
                break

        return done

    def dino_target_detection(self, img, depth, object_info):
        target_detections = []
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops

        img_src = copy.deepcopy(np.array(img))
        img = Image.fromarray(img_src)
        prompt = object_info
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_transformed, _ = transform(img, None)
        # image_transformed, _ = transform(img)
        boxes, logits, phrases = self.dino_model(image=image_transformed,
                                                 caption=prompt,
                                                 box_threshold=0.6,
                                                 text_threshold=0.40)
        logits = logits.detach().cpu().numpy()
        H, W, _ = img_src.shape
        boxes_xyxy = (box_ops.box_cxcywh_to_xyxy(boxes) *
                      torch.Tensor([W, H, W, H])).cpu().numpy()
        boxes = []
        for box in boxes_xyxy:
            if (box[2] - box[0]) / W > 0.6 or (box[3] - box[1]) / H > 0.5:
                continue
            boxes.append(box)

        if len(boxes) > 0:
            for i, point in enumerate(boxes):
                point = list(map(int, point))
                center_point = (int(
                    (point[0] + point[2]) / 2), int((point[1] + point[3]) / 2))
                depth_data = int(depth[center_point[1], center_point[0]] /
                                 2.55)
                target_detections.append((float(logits[i]), depth_data))

        return len(target_detections) > 0
