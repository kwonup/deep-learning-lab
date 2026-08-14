#라벨만들기 위한 os, json
import matplotlib.pyplot as plt
import os
import numpy as np #비전 관련 작업 cv로 진행할때는,numpy를 함께 임포트
import cv2
from ultralytics import YOLO
import shutil #.sh/ .bash

from utils import augmentation as aug
import albumentations as A

from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights

def count_params(model):
    #requires_grad = True (훈련시킬것,변경가능) = False(훈련안시킴,변경불가)
    #p.numel(파라미터의 구성요소 개수)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'전체 파라미터  : {total:,}')
    print(f'훈련 가능 파라미터  :  {trainable:,}')
    print(f'동결된 파라미터  : {total-trainable:,}')


if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    #Faster RCNN
    #Backbone => 이미지의 특징 추출 => 저수준의 특징 추출(Resnet)
    #RPN(Resion Proposal Networks) => Bounding Box의 후보 제안
    #ROI Head => RPN을 본 뒤 , 분류 수행, bbox 보정**
    #Faster RCNN: 클래스 = 분류하고자 하는 객체의 개수

    print(model)

    in_featrues = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_head.box_predictor = FasterRCNNPredict(in_featrues,num_classes=2)


    

#딥러닝 시퀀스
#1. 데이터 가져옴
#2. 데이터 정제(preprocessing)
#3. 알고리즘 선택
#4. 훈련 
#5. 검증
#6. 평가
#7. 배포