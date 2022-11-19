
import gradio as gr

import os
import argparse
import cv2
import time
from predictor import Predictor
from ssd.config import cfg
def get_parser():
    parser = argparse.ArgumentParser(description="SSD gradio demo")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--score_threshold", type=float, default=0.2)
    parser.add_argument("--model_number", default=2, choices=[0 ,1 ,2, 3], type=str)
    return parser.parse_args()

args = get_parser()
if args.model_number == 0:
    cfg.merge_from_file('./configs/vgg_ssd300_voc0712.yaml')
elif args.model_number == 1:
    cfg.merge_from_file('./configs/vgg_ssd512_voc0712.yaml')
elif args.model_number == 2:
    cfg.merge_from_file('./configs/vgg_ssd300_coco_trainval35k.yaml')
elif args.model_number == 3:
    cfg.merge_from_file('./configs/vgg_ssd512_coco_trainval35k.yaml')
else:
    raise OSError('wrong model_number, set one in [0 , 1 , 2, 3]')
cfg.freeze()
## 加载
predictor = Predictor(cfg=cfg, args=args)

def SSD(video_path):
    print('gheuwgahlewg-', video_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频的平均帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频格式
    temp_path = './temp.mp4'  # 视频暂存路径
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

  
title = "SSD"
description = "SSD的小demo，参考自SSD github，请查看下述地址。"
article = "<a href='https://github.com/lufficc/SSD'>Github</a></p>"

demo = gr.Interface(
    fn=SSD,
    inputs=gr.Video(source='upload'),
    outputs=gr.Video(),
    title=title,
    description=description,
    article=article,
    examples=[["./palace.mp4"]]
)
demo.launch(debug=True, share=False)
