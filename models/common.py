import math

import torch
import torch.nn as nn


"""
- Conv
- Bottleneck
- SPP
- DWConv
- Focus
- BottleneckCSP
- Concat
- NMS
"""

"""
k(kernel)
p(padding)
"""
def autopad(k,p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k,int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):

    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,act=True):
        super(Conv,self).__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,autopad(k,p),groups=g,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
    
    def fuseforward(self,x):
        return self.act(self.conv(x))

"""
g(groups)
e(expansion)
"""
class Bottleneck(nn.Module):

    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super(Bottleneck,self).__init__()
        # hidden channels 
        
        c_ = int(c2 * e)
        # [8,416,416]
        self.cv1 = Conv(c1,c_,1,1)
        # [16,416,416]
        self.cv2 = Conv(c_,c2,3,1,g=g)
        self.add = shortcut and c1 == c2

    def forward(self,x):
        # 
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

"""
n(number)
"""
class BottleneckCSP(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=True,g=1,e=0.5):
        super(BottleneckCSP,self).__init__()
        c_ = int(c2 * e)
        # input [16,416,416] output [8,416,416]
        self.cv1 = Conv(c1,c_,1,1)
        # input [16,416,416] output [8,416,416]
        self.cv2 = nn.Conv2d(c1,c_,1,1,bias=False)
        # input [8,416,416] output [8,416,416]
        self.cv3 = nn.Conv2d(c_,c_,1,1,bias=False)
        # input [16,416,416] output [16,416,416]
        self.cv4 = Conv(2*c_,c2,1,1)
        # 16
        self.bn = nn.BatchNorm2d(2*c_)

        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_,c_,shortcut,g,e=1.0) for _ in range(n)])


    def forward(self,x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1,y2),dim=1))))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    # 特征金字塔结构
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # 
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1)) 

# 全局注意力
# [3,416,416]
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,c1,c2,k=1,p=None,g=1,act=True):
        # 3 * 64 = 
        # 256 3 416//2 = 208
        self.conv = Conv(c1 * 4,c2,k,s,p,g,act)
    def forward(self,x):# (b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat(x[]))

if __name__ == "__main__":

    # test conv
    # conv = Conv(3,16,k=3)
    # img = torch.randint(0,255,(1,3,416,416),dtype=torch.float32)
    # res = conv(img)
    # print(res.shape)

    # test bottleneck
    # conv = Bottleneck(16,16)
    # img = torch.randint(0,255,(1,16,416,416),dtype=torch.float32)
    # res = conv(img)
    # print(res.shape)
    
    # test BottleneckCSP
    # conv = BottleneckCSP(16,16,n=3)
    # img = torch.randint(0,255,(1,16,416,416),dtype=torch.float32)
    # res = conv(img)
    # print(res.shape)

    # Focus
    

    