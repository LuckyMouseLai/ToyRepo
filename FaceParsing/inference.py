
import cv2
import torch
import numpy as np
from skimage.filters import gaussian
from torchvision import transforms
import onnxruntime

from network import BiSeNet

## 属性
atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
class Inference():
    def __init__(self, device, use_onnx) -> None:
        self.device = device
        self.use_onnx = use_onnx
        if device == 'cpu':
            self.session = onnxruntime.InferenceSession('./checkpoints/79999_iter.onnx', providers=['CPUExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession('./checkpoints/79999_iter.onnx', providers=['CUDAExecutionProvider'])

        self.model = BiSeNet(n_classes=19).to(self.device)
        self.model.load_state_dict(torch.load('./checkpoints/79999_iter.pth'))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    ## 图像预处理，输入BGR图
    def preprocess(self, image):
        image_BGR = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image)
        image = torch.unsqueeze(image, 0)
        return image, image_BGR
    
    ## 显示各个部位的分割结果
    def vis_parsing_maps(self, image, parsing_anno, stride=1):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 0, 85], [255, 0, 170],
                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
                    [0, 255, 85], [0, 255, 170],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [0, 85, 255], [0, 170, 255],
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        image = np.array(image)
        vis_im = image.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)  # 查找索引
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]  # 根据索引上色

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)  # 上色后的图
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)  # 将原图和上色后的图按权重叠加

        return vis_parsing_anno_color, vis_im  # 脸部标注图 标注原图叠加，

    def sharpen(self, img):
        img = img * 1.0
        gauss_out = gaussian(img, sigma=5, multichannel=True)

        alpha = 1.5
        img_out = (img - gauss_out) * alpha + img

        img_out = img_out / 255.0

        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2
        img_out = np.clip(img_out, 0, 1)
        img_out = img_out * 255
        return np.array(img_out, dtype=np.uint8)

    def hair(self, image, parsing, part=17, color=[230, 50, 20]):
        b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
        tar_color = np.zeros_like(image)  # 将每个通道的像素值都改为目标color的像素值
        tar_color[:, :, 0] = b  
        tar_color[:, :, 1] = g
        tar_color[:, :, 2] = r

        ## 为什么转为hsv域，H:色调(0-360度)， S：饱和度(0-100%) V：明度(0-100%)
        ## opencv的hsv取值分别为(0-179, 0-255, 0-255)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 原图转hsv
        tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)  # 目标颜色图转hsv

        ## 嘴唇hs变换, 头发h变换
        if part == 12 or part == 13:
            image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]  ## 修改hs
        else:
            image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]  ## 修改h

        changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        ## 头发颜色，在BGR域中变换
        if part == 17:
            changed = self.sharpen(changed)
        changed[parsing != part] = image[parsing != part]  # 修改像素

        return changed

    def mosaic(self, image, parsing):
        
        pass
    
    ## image: BGR
    def inference(self, image, mode):
        with torch.no_grad():
            self.model.eval()
            input, image_BGR = self.preprocess(image=image)  # input：1*3*512*512， image_BGR: 512*512*3
            if self.use_onnx:
                output = self.session.run(['output_0'], input_feed={'input_0': input.numpy()})
                output = torch.tensor(output[0])
            else:
                output = self.model(input.to(self.device))  # output: 1*19*512*512   19为类别数，下面取每个位置上的最大值
            parsing = output.squeeze(0).cpu().numpy().argmax(0)  # parsing: (512* 512) 
            ### 根据mode返回不同的图片
            if mode == 0:
                visual_image = self.vis_parsing_maps(image_BGR, parsing)[0]
            elif mode == 1:
                visual_image = self.vis_parsing_maps(image_BGR, parsing)[1]
            elif mode == 2:
                table = {
                    'hair': 17,
                    'upper_lip': 12,
                    'lower_lip': 13
                }
                parts = [table['hair'], table['upper_lip'], table['lower_lip']]  # 修改部位
                # parts = [table['upper_lip'], table['lower_lip']]  # 修改部位
                colors = [[155, 20, 20], [150, 80, 180], [150, 80, 180]]  # 目标颜色
                # colors = [[20, 70, 180], [20, 70, 180], [20, 70, 180]]  # 目标颜色

                for part, color in zip(parts, colors):
                    image_BGR = self.hair(image_BGR, parsing, part, color)
                visual_image = image_BGR
            visual_image = cv2.resize(visual_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return visual_image

