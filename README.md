# m_yolov5
yolov5 with pytorch

创建 models 模块，并创建 yolo.py 文件


```python
device = select_device(cfg)
model = Model(cfg).to(device)
```

配置日志输出格式以及级别
```python

def set_logging(rank=-1):
    logging.basicConfig(
            format="%(message)s",
            level=logging.INFO if rank in [-1,0] else logging.WARN)

```
创建 utils 并实现 select_device 方法

```python
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
```

### 基础网络结构快
- Conv
```python
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


if __name__ == "__main__":
    conv = Conv(3,16,k=3)
    img = torch.randint(0,255,(1,3,416,416),dtype=torch.float32)
    res = conv(img)
    print(res.shape)
```
- Bottleneck
```python
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
```

- SPP
- DWConv
- Focus
- BottleneckCSP
- Concat
- NMS

#### hard-swish

$$ f(x) = x \times sigmoid(\beta x) $$

$$ f^{\prime}(x) = 1 \times sigmoid (\beta x) + x(\beta \times sigmoid(\beta x))$$

- $\beta = 0$ 时 $f(x) = \frac{x}{2}$
- $\beta$ 趋近于无穷 $\sigma(x) = (1 + \exp(-x))- \sigma(x) = (1 + \exp(-x))-1 $ 为 0 或 1 ，swish 变为 ReLU $f(x) = 2 \max(0,x)$
所以Swish函数可以看做是介于线性函数与ReLU函数之间的平滑函数. beta是个常数或者可以训练的参数。其具有无上界有下界、平滑、非单调的特性。其在模型效果上优于ReLU。
hard-Swish 介绍
虽然 swish 非线性提高了精度，但是在嵌入式环境中，成本是非零的，因为在移动设备上计算 sigmoid 函数代价要大得多。

$$h-swish(x) = x \frac{ReLU6(x+3)}{6}$$

### 运行时熔断 BN 和 Conv

我们讨论如何通过合并冻结批处理规范化层和前面的卷积来简化网络结构。这是实践中常见的设置，值得研究。批处理规范化（通常缩写为BN）是现代神经网络中常用的一种方法，因为通常可以减少训练时间，并有可能提高泛化能力（但是，围绕它有一些争议：1，2）。

[fusing-batchnorm-and-conv](https://nenadmarkus.com/p/fusing-batchnorm-and-conv/)