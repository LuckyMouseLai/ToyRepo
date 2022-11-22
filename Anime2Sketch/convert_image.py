
import cv2
import argparse
import os

from inference import Inference

def get_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    parse.add_argument('--path', type=str, default='./images/madoka.jpg')
    parse.add_argument('--use_onnx', type=bool, default=False)
    parse.add_argument('--save_path', type=str, default=None, help='img save path, it should be same with input format')

    return parse.parse_args()

if __name__ == '__main__':
    args = get_parser()
    predictor = Inference(device=args.device, use_onnx=args.use_onnx)
    image = cv2.imread(args.path)  # BGR image
    new_frame = predictor.inference(image=image)
    if args.save_path is None:
        texts = os.path.splitext(args.path)
        save_path =texts[0]+'_sketch'+ texts[1]
    else:
        save_path = args.save_path
    cv2.imwrite(save_path, new_frame)

