import os

import torch
import logging

from common import *

logger = logging.getLogger(__name__)


def select_device(device='',batch_size=None):
    # device = 'cpu' or 'number'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:#如果使用 gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # 检测 cuda 是否可用
        assert torch.cuda.is_available(), 'CUDA unavailable invalid device %s requested' % device 
    
    cuda = False if cpu_request else torch.cuda.is_available()

    if cuda:
        # bytes to MB
        c = 1024**2
        
        ng = torch.cuda.device_count()
        # 验证 batch_size 是否与 device_count 匹配
        # 如果确保 6 位有效数字前提下，使用小数方式，否则使用科学计数法
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, 'batch_size %g not multiple of GPU count %g' %(batch_size,ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA'
        for i in range(0,ng):
            if i == 1:
                s = ' '* len(s)
            logging.info("%sdevice%g _CudaDeviceProperties(name='%s',total_memory=%dMB)"%
                (s,i,x[i].name,x[i].total_memory/c))
    else:
        logging.info('Using CPU')

    logging.info('')
    return torch.device('cuda:0'if cuda else 'cpu')

if __name__ == "__main__":
    set_logging()
    logging.info('hello')
    
    device = select_device('1')
    print(device)