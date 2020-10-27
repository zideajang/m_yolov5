
import math
import logging


import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
    
    def forward(self,x):
        pass
    def forward_once(self,x):
        pass
    def _initialize_biases(self):#在初始化 cf 是 class frequence
        pass
    def _print_biases(self):
        pass
    def fuse(self):
        pass
    def add_nms(self):#
        pass
    def info(self):#输出模型信息
        pass
if __name__ == "__main__":
    print("hello yolov5...")



    # device = select_device(cfg)
    # model = Model(cfg).to(device)