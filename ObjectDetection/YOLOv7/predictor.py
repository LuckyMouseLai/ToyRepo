import cv2
import torch
import time
import numpy as np
from vizer.draw import draw_boxes
import onnxruntime

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, select_device

class Predictor():
    def __init__(self, args) -> None:
        self.device = select_device(args.device)
        self.model = attempt_load('./checkpoints/yolov7.pt', map_location=self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model = self.model.half()
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(args.image_size, s=self.stride)  # check img_size
            # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 80类
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]  # 随机选择一个颜色框
        self.iou_thre = args.iou_threshold
        self.conf_thre = args.conf_threshold
        
    @ torch.no_grad()
    def inference_video(self, frame):
        image = letterbox(frame, self.imgsz, self.stride)[0]
        ## Covert
        image = image[:,:,::-1].transpose(2,0,1)  # BGR2RGB
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255.0
        image = image.unsqueeze(0)
        pred = self.model(image, augment=False)[0]  # augment: TTA?
        pred = non_max_suppression(pred, conf_thres=self.conf_thre, iou_thres=self.iou_thre)
        for det in pred:
            s = ''
            if len(det):
                # Rescale boxes from img_size to image_BGR size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], frame.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=self.colors[int(cls)], line_thickness=1)
        return frame
   
    @ torch.no_grad()
    def inference_img(self, image_path):
        t0 = time_synchronized()
        image_BGR = cv2.imread(image_path)
        assert image_BGR is not None, "Image Not Found"
        image = letterbox(image_BGR, self.imgsz, self.stride)[0]
        ## Covert
        image = image[:,:,::-1].transpose(2,0,1)  # BGR2RGB
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255.0
        image = image.unsqueeze(0)
        t1 = time_synchronized()
        pred = self.model(image, augment=False)[0]  # augment: TTA?
        t2 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres=self.conf_thre, iou_thres=self.iou_thre)
        t3 = time_synchronized()
        for det in pred:
            s = ''
            if len(det):
                # Rescale boxes from img_size to image_BGR size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_BGR.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, image_BGR, label=label, color=self.colors[int(cls)], line_thickness=1)
        print(f'{s}Done. ({(1E3 * (t1 - t0)):.1f}ms) Preprocess, ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        print(f'Done. ({time.time() - t0:.3f}s)')
        return image_BGR
        
class OnnxPredictor():
    def __init__(self, args) -> None:
        self.device = select_device(args.device)
        self.model = attempt_load('./checkpoints/yolov7.pt', map_location=self.device)
        self.session = onnxruntime.InferenceSession('./checkpoints/yolov7.onnx', providers=['CPUExecutionProvider'])
        # self.session = onnxruntime.InferenceSession('./checkpoints/yolov7.onnx', providers=['CUDAExecutionProvider'])
    
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model = self.model.half()
        # self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(args.image_size, s=self.stride)  # check img_size
            # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 80类
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]  # 随机选择一个颜色框
        self.iou_thre = args.iou_threshold
        self.conf_thre = args.conf_threshold
        
    @ torch.no_grad()
    def inference_video(self, frame):
        image = letterbox(frame, self.imgsz, self.stride)[0]
        ## Covert
        image = image[:,:,::-1].transpose(2,0,1)  # BGR2RGB
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255.0
        image = image.unsqueeze(0)
        pred = self.session.run(output_names=['output_0'], input_feed={'input_0': image.numpy()})
        pred = torch.tensor(pred[0])
        # pred = self.model(image, augment=False)[0]  # augment: TTA?
        pred = non_max_suppression(pred, conf_thres=self.conf_thre, iou_thres=self.iou_thre)
        for det in pred:
            s = ''
            if len(det):
                # Rescale boxes from img_size to image_BGR size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], frame.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=self.colors[int(cls)], line_thickness=1)
        return frame
   
    @ torch.no_grad()
    def inference_img(self, image_path):
        t0 = time_synchronized()
        image_BGR = cv2.imread(image_path)
        assert image_BGR is not None, "Image Not Found"
        image = letterbox(image_BGR, self.imgsz, self.stride)[0]
        ## Covert
        image = image[:,:,::-1].transpose(2,0,1)  # BGR2RGB
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()  # uint8 to fp16/32
        image /= 255.0
        image = image.unsqueeze(0)
        t1 = time_synchronized()
        pred = self.session.run(['output_0'], input_feed={'input_0': image.numpy()})
        pred = torch.tensor(pred[0])
        # pred = self.model(image, augment=False)[0]  # augment: TTA?
        t2 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres=self.conf_thre, iou_thres=self.iou_thre)
        t3 = time_synchronized()
        for det in pred:
            s = ''
            if len(det):
                # Rescale boxes from img_size to image_BGR size
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_BGR.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, image_BGR, label=label, color=self.colors[int(cls)], line_thickness=1)
        print(f'{s}Done. ({(1E3 * (t1 - t0)):.1f}ms) Preprocess, ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        print(f'Done. ({time.time() - t0:.3f}s)')
        return image_BGR
 

