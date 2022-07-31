
import cv2
import torch
from torchvision import transforms

from network import BiSeNet

## 属性
atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
class Inference():
    def __init__(self) -> None:
        self.model = BiSeNet(n_classes=19).cuda() 
        self.model.load_state_dict(torch.load('./checkpoints/79999_iter.pth'))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def preprocess(self, image):
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.to_tensor(image)
        image = torch.unsqueeze(image, 0)
        return image

    def inference(self, image):
        with torch.no_grad():
            self.model.eval()
            image = self.preprocess(image=image)
            output = self.model(image.cuda())[0]  # output: 1*19*512*512   19为类别数，下面取每个位置上的最大值
            parsing = output.squeeze(0).cpu().numpy().argmax(0)  # parsing: (512* 512) 
        # cv2.imwrite('./test.png', parsing)

