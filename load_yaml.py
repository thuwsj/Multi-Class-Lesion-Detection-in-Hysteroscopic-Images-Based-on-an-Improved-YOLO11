# -*- coding:utf-8 -*-
from ultralytics import YOLO
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# #
# def main():
#     yaml_path = r'tiny_backbone/yolov11_regnety_002.yaml'
#     model = YOLO(yaml_path,verbose=True)
#     # print(model)
#
# if __name__ == '__main__':
#     main()



import timm
for model in timm.list_models():
    print(model)

