import os
import time
import argparse

from PIL import Image

from ssd.config import cfg
from predictor import Predictor

def get_parser():
    parser = argparse.ArgumentParser(description="SSD image inference demo")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--image_path", default='./images/003123.jpg', type=str, help='Specify a image path to do prediction.')
    parser.add_argument("--model_number", default=0, choices=[0 ,1 ,2, 3], type=str)

    return parser.parse_args()


def main():
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
    image_path = args.image_path
    predictor = Predictor(cfg=cfg, args=args)
    image_name = os.path.basename(image_path)
    texts = os.path.splitext(image_path)
    savepath =texts[0]+'_result'+ texts[1]
    ## 推理
    start = time.time()
    drawn_image, message = predictor.inference_img(image_path)  # 结果图，目标个数信息
    inference_time = time.time() - start

    print(image_name, message, 'inference {:03d}ms'.format(round(inference_time * 1000)))
    Image.fromarray(drawn_image).save(savepath)


if __name__ == '__main__':
    main()