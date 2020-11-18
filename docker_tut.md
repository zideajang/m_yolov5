基于 GPU 安装 Pytorch 1.5 docker Ubuntu 20.04 LTS
```
sudo apt install docker-compose
```

### 安装兼容版本的 gcc
在 Ubuntu 20.04 LTS 中使用的 gcc 版本为 9，所以首先需要安装兼容的版本 gcc install
```
sudo apt -y install build-essential
```
```
sudo apt -y install gcc-8 g++-8 gcc-9 g++-g
```
`update-alternatives` 命令用于处理 Linux 系统中软件版本的切换，使其多版本共存。

```
docker version
```