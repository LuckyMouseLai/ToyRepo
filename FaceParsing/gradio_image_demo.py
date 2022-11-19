
import gradio as gr

import cv2
import torch
import numpy as np
from torchvision import transforms
import onnxruntime
import imageio

from inference import Inference

## 单张图片转换
predictor = Inference(device='cpu', use_onnx=True)
def face_parsing(image_path, mode):
    image = cv2.imread(image_path)
    image = predictor.inference(image=image, mode=mode)
    # cv2.imwrite(temp_path, image)
    image = image[:,:,::-1]
    return image


title = "FaceParsing"
description = "demo for FaceParsing. To use it, simply upload your image, or click one of the examples to load them."

demo = gr.Interface(
    fn=face_parsing,
    inputs=[gr.Image(type='filepath', source='upload'), gr.Dropdown(choices=['人脸解析1','人脸解析2','修改口红发色'], type='index')],
    outputs=gr.Image(),
    title=title,
    description=description,
    examples=[['./results/qiushuzhen.png']]
)
demo.launch(debug=False, share=False)
