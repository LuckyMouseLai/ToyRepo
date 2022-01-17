import cv2
from matplotlib import pyplot as plt
import glob
import numpy as np
import os
import csv
from dlib_tool import recognizer, detector, predictor
path_to_features_csv = "./data/face_features.csv"

# 获取人脸的128D特征
def get_face_features(img_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)
    
    if len(faces) != 0:
        shape = predictor(img_rgb, faces[0])
        face_featture = recognizer.compute_face_descriptor(img_rgb, shape)
    else:
        face_featture = 0  ## 无人脸
    return face_featture

def save_features(face_name, face_dir):
    assert os.path.exists(face_dir), '目录不存在'
    assert os.path.isdir(face_dir), '请输入一个目录'

    features_list = []
    file = glob.glob(face_dir+'/*')
    print(file)
    for i in file:
        face_featture = get_face_features(i)
        # 遇到没有检测出人脸的图片跳过
        if face_featture == 0:
            print('no face')
        else:
            features_list.append(face_featture)
    with open(path_to_features_csv, "a") as csvfile:
        writer = csv.writer(csvfile)
        features_mean = np.array(features_list).mean(axis=0)
        info = list(features_mean)
        info.insert(0, face_name)
        writer.writerow(info)
# save_features('周润发', './face/zrf')