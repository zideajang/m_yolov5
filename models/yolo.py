
import math
import logging


import torch
import torch.nn as nn
import yaml
from pathlib import Path

class Model(nn.Module):
    """
    ch=3
    nc(class number)
    """
    def __init__(self,cfg='yolov5s.yaml', ch=3, nc=None):
        super(Model,self).__init__()
        # 读取配置文件
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader) 

        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value

    
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

    model = Model()

    # device = select_device(cfg)
    # model = Model(cfg).to(device)