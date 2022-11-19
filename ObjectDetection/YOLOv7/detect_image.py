import os
import argparse
import cv2

from predictor import Predictor

def get_parser():
    parser = argparse.ArgumentParser(description="YOLOv7 image inference demo")
    parser.add_argument("--device", type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.45)
    parser.add_argument("--image_path", default='./horses.jpg', type=str, help='Specify a image path to do prediction.')
    parser.add_argument("--image_size", default=640, type=int, help='inference size')

    return parser.parse_args()


def main():
    args = get_parser()
    image_path = args.image_path
    predictor = Predictor(args=args)
    image_name = os.path.basename(image_path)
    texts = os.path.splitext(image_path)
    savepath =texts[0]+'_result'+ texts[1]
    ## 推理
    drawn_image = predictor.inference_img(image_path)  # 结果图，目标个数信息
    cv2.imwrite(savepath, drawn_image)
    print('image has been save in', savepath)

if __name__ == '__main__':
    main()