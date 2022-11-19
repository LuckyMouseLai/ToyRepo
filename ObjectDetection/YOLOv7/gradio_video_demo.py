
import gradio as gr

import os
import argparse
import cv2
import time
from predictor import OnnxPredictor

def get_parser():
    parser = argparse.ArgumentParser(description="YOLO image inference demo")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.45)
    parser.add_argument("--image_size", default=640, type=int, help='inference size')

    return parser.parse_args()
## 加载
predictor = OnnxPredictor(args=get_parser())

def yolov7(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频的平均帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频格式
    input_format = video_path.split('.')[-1]  # 输入视频格式
    temp_path = './temp.'+input_format  # 视频暂存路径
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))  # 视频写入对象
    if not cap.isOpened():  # 判断视频流是否打开
        print('Failed to read frame')
        exit()
    while True:
        flag, frame = cap.read()
        if not flag:
            print('Failed to read frame')
            writer.release()
            cap.release()
            break
        new_frame = predictor.inference_video(frame)
        writer.write(new_frame)
    return temp_path  # 返回暂存视频路径

  
title = "YOLO v7"
description = "YOLO v7的小demo，参考自yolov7论文和官方github，请查看下述地址。"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2207.02696'>YOLO v7论文</a> | <a href='https://github.com/WongKinYiu/yolov7'>官方Github</a></p>"

demo = gr.Interface(
    fn=yolov7,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title=title,
    description=description,
    article=article,
    examples=[["./palace.mp4"]]
)
demo.launch(debug=True, share=True)
