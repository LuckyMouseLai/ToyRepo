
import cv2
import torch
import numpy as np
from torchvision import transforms
import onnxruntime

from network.model import create_model


class Inference():
    def __init__(self, device, use_onnx) -> None:
        self.device = device
        self.use_onnx = use_onnx
        if device == 'cpu':
            self.session = onnxruntime.InferenceSession('./checkpoints/netG.onnx', providers=['CPUExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession('./checkpoints/netG.onnx', providers=['CUDAExecutionProvider'])
        self.model = create_model('./checkpoints/netG.pth').to(self.device)
        # self.model = BiSeNet(n_classes=19).to(self.device)
        # self.model.load_state_dict(torch.load('./checkpoints/netG.pth'))
        self.to_tensor = transforms.Compose([
            # transforms.Resize(512, 512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    ## 图像预处理，输入BGR图
    def preprocess(self, image):
        image_BGR = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image)
        image = torch.unsqueeze(image, 0)
        return image, image_BGR
    
    ## image: BGR
    def inference(self, image):
        with torch.no_grad():
            self.model.eval()
            input, image_BGR = self.preprocess(image=image)  # input：1*3*512*512， image_BGR: 512*512*3
            if self.use_onnx:
                output = self.session.run(['output_0'], input_feed={'input_0': input.numpy()})
                output = torch.tensor(output[0])
            else:
                output = self.model(input.to(self.device))  # output: 1*19*512*512   19为类别数，下面取每个位置上的最大值
            output = output.squeeze(0).cpu().numpy()  # parsing: (512* 512) 
            output = (np.transpose(output, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            output = output.astype(np.uint8)
            output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)  # resize回原来大小
        return output

