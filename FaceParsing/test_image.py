import string
import cv2
import argparse
import os

from matplotlib.pyplot import text

from inference import Inference

def get_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    parse.add_argument('--mode', type=int, default=2, choices=[0, 1, 2, 3, 4])  # 0: 标注 1: 标注+原图融合 2：
    parse.add_argument('--path', type=str, default='./results/liuxing.png')
    return parse.parse_args()


if __name__ == '__main__':
    args = get_parser()
    predictor = Inference(device=args.device)
    image = cv2.imread(args.path)  # BGR image
    new_frame = predictor.inference(image=image, mode=args.mode)
    texts = os.path.splitext(args.path)
    savepath =texts[0]+'_mode{}'.format(args.mode) + texts[1]
    cv2.imwrite(savepath, new_frame)

