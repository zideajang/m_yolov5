# m_yolov5
yolov5 with pytorch

创建 models 模块，并创建 yolo.py 文件


```python
device = select_device(cfg)
model = Model(cfg).to(device)
```


```python

def set_logging(rank=-1):
    logging.basicConfig(
            format="%(message)s%",
            level=logging.INFO if rank in [-1,0] else logging.WARN)

```
创建 utils 