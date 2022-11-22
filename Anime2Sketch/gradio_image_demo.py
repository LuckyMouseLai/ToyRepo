
import gradio as gr

import cv2
import torch
import numpy as np
from torchvision import transforms
import onnxruntime


## 加载onnx会话
session = onnxruntime.InferenceSession('./checkpoints/netG.onnx', providers=['CPUExecutionProvider'])
## to tensor
to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
## 数据预处理
def preprocess(image):
    image_BGR = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    image = to_tensor(image)
    image = torch.unsqueeze(image, 0)
    return image
## 单张图片转换
def inference(image):
    input = preprocess(image=image)
    output = session.run(['output_0'], input_feed={'input_0': input.numpy()})
    output = torch.tensor(output[0])
    output = output.squeeze(0).cpu().numpy()  # parsing: (512* 512) 
    output = (np.transpose(output, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    output = output.astype(np.uint8)
    output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)  # resize回原来大小
    return output

def sketch2anime(image_path):
    input_format = image_path.split('.')[-1]  # 输入图片格式
    temp_path = './images/temp.'+input_format  # 视频暂存路径
    image = cv2.imread(image_path)
    image = inference(image)
    cv2.imwrite(temp_path, image)
    return temp_path


title = "Anime2Sketch"
description = "demo for Anime2Sketch. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"

demo = gr.Interface(
    fn=sketch2anime,
    inputs=gr.Image(type='filepath', source='upload'),
    # inputs=gr.Image(type='filepath', source='webcam'),
    # inputs=gr.Image(type='filepath', source='canvas'),
    outputs=gr.Image(type='filepath'),
    title=title,
    description=description,
    article=article,
    examples=[['images/aa.jpg'],['./images/Rengoku_Kyoujurou.jpg'], ['./images/nidemingzi4.jpg']]
)
demo.launch(debug=False, share=False)
