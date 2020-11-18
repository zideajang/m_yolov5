
import math
import logging

from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from pathlib import Path

# from utils.general import *

def set_logging(rank=-1):
    logging.basicConfig(
            format="%(message)s",
            level=logging.INFO if rank in [-1,0] else logging.WARN)

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
        self.model,self.save = parse_model(
            
        )

    
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
"""
d model_dict
ch channel
"""
def parse_model(d,ch):
    logging.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'],d['nc'],d['depth_multiple'],d['width_multiple']
    logging.info(f'anchors:{anchors}\n nc:{nc}')
    # number of anchors
    na = (len(anchors[0]) // 2) if isinstance(anchors,list) else anchors
    # number of outputs = anchors  * (classes + 5)
    no = na * (nc + 5)
    logging.info(f'na:{na}, no:{no}, nc:{nc}')

    logging.info(f'backbone: {d["backbone"]}')
    logging.info('\n')
    logging.info(f'head: {d["head"]}')
    layers, save, c2 = [],[], ch[-1]


    for i,(f,n,m,args) in enumerate(d['backbone'] + d['head']):
        logging.info('--------- layer start --------')
        logging.info(f'f: {f}')
        logging.info(f'n: {n}')
        logging.info(f'm: {m}')
        logging.info(f'args: {args}')
        logging.info('--------- layer end --------')
        # eval 将 m 转换为表达式
        m = eval(m) if isinstance(m,str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a,str) else a 
            except:
                pass
        n = max(round(n * gd))

    return nn.Sequential(*layers), sorted(save)

if __name__ == "__main__":
    # print("hello yolov5...")
    set_logging()
    with open("yolov5s.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) 
        parse_model(cfg,[3])
    # model = Model()

    # device = select_device(cfg)
    # model = Model(cfg).to(device)