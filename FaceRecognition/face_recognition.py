import cv2
import numpy as np
import os
import pandas as pd

from dlib_tool import predictor, detector, recognizer

path_to_features_csv = "./data/face_features.csv"

# 计算2个向量的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    distance = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return distance

def face_rec(img_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 1)  ## 检测人脸位置
    if len(faces)==0:
        print('未检测到人脸')
        return 
    shape = predictor(img_rgb, faces[0])  ## 如果检测到多个人脸，取第一张人脸
    face_featture = recognizer.compute_face_descriptor(img_rgb, shape)

    ###
    face_df = pd.read_csv(path_to_features_csv, header=None)
    distance_list = []
    for i in range(face_df.shape[0]):
        name = face_df.loc[i][0]
        feature = face_df.loc[i][1:]
        distance = return_euclidean_distance(face_featture, feature)  ## 待识别的人脸特征与库中的人脸特征 的距离计算
        distance_list.append(distance)
    similar_person_index = distance_list.index(min(distance_list))
    ### 取最小距离，当距离小于一个阈值(可以修改)时，我们判定他们是同一个人
    if min(distance_list) < 0.4:
        print('name:', face_df.loc[similar_person_index][0], 'prob:', 1-min(distance_list))
    else:
        print("Unknown person")

face_rec('./wbb.jpg')