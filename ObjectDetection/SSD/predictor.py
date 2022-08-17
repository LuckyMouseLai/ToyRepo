from email import message
import torch
import cv2
import numpy as np
from vizer.draw import draw_boxes

from PIL import Image
from ssd.data.datasets import COCODataset, VOCDataset
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model

class Predictor():
    def __init__(self, cfg, args) -> None:
        if args.model_number ==0:
            self.class_names = VOCDataset.class_names
            model_path = './checkpoints/vgg_ssd300_voc0712.pth'
        elif args.model_number ==1:
            self.class_names = VOCDataset.class_names
            model_path = './checkpoints/vgg_ssd512_voc0712.pth'
        elif args.model_number ==2:
            self.class_names = COCODataset.class_names
            model_path = './checkpoints/vgg_ssd300_coco_trainval35k.pth'
        elif args.model_number ==3:
            self.class_names = COCODataset.class_names
            model_path = './checkpoints/vgg_ssd512_coco_trainval35k.pth'


        self.device = args.device
        self.score_threshold = args.score_threshold
        model = build_detection_model(cfg)
        model = model.to(self.device)

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))    
        model.load_state_dict(state_dict['model'])
        self.model = model
        self.model.eval()
        self.to_tensor = build_transforms(cfg, is_train=False)
    @ torch.no_grad()
    def inference(self, rgb_image):
        images = self.to_tensor(rgb_image)[0].unsqueeze(0)
        result = self.model(images.to(self.device))[0]
        return result
    def inference_video(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        result = self.inference(image)
        result = result.resize((width, height)).to('cpu').numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        indices = scores > self.score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        drawn_image = draw_boxes(frame, boxes, labels, scores, self.class_names).astype(np.uint8)
        return drawn_image

    def inference_img(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        result = self.inference(image)
        result = result.resize((width, height)).to('cpu').numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        indices = scores > self.score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        drawn_image = draw_boxes(image, boxes, labels, scores, self.class_names).astype(np.uint8)
        message = 'detect {:d} objects'.format(len(boxes))
            
        
        return drawn_image, message
        

