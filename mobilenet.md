### MobileNet 模型
MobileNet 模型是 google 在 2017 年针对手机或者嵌入式提出轻量级模型。提出 MobileNet 这样高效模型的是 google 的 Andrew G.Howard, Menglong Zhu 等人。


#### 背景
之前介绍的 AlexNet 到 ResNet 这些神经网络结构为了得到更高准确度，更倾向于把网络做的更深，更复杂，可想而知这样网络结构势必带来大量参数，消耗大量计算资源，但是考虑到一些需要性能和速度，而且设备资源有限情况下，我们需要一个准确度可以接受，效率高网络结构，这就是 MobileNet 由来的动机。
- 自动驾驶
- 移动设备上目标运行识别

![mobilenet_001.jpg](https://upload-images.jianshu.io/upload_images/8207483-24e7d592bb2eb477.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


自从 2016 年，大家更专注如何将模型应用到产品，也就是从讲故事做 demo 阶段来到了如何将其应用商用上，这样也就是必须考虑性能，可行性。

- SqueezeNet
- MobileNetV1 ShuffleNet
- MobileNetV2 NASNet
- MobileNetV3 MnasNet

接下来通过具体数据来说明 mobileNet 相对其他网络结构其特点，在 ImageNet 数据集上，对比 VGG16 在参数减少了 30 多倍情况下，准确率与 VGG16 只相差了 0.9%

#### 实现方法
- 压缩模型: 在已经训练好的模型上进行压缩，使得网络携带更少的网络参数
- 设计轻量级模型: 设计出更高效的网络计算方式来减少网络参数

#### 主体结构
![mobilenet_002.jpg](https://upload-images.jianshu.io/upload_images/8207483-0adebc4ef61b2fa8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- Conv: 表示标准卷积
- s2: 卷积的步长 stride 为 2
- Avg Pool:平均池化
- FC: 全连
- dw: 深度卷积

将 DW 卷积和 PW 卷积看作两层，一共 28 层。
##### 卷积块
引入了深度可分离卷积将卷积替换为 DW 卷积和 PW 卷积的组合。

![imagenet_003.jpeg](https://upload-images.jianshu.io/upload_images/8207483-2d26726a3bdd4058.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这里看右侧的卷积块结构，在深度可分离卷积并不是 3 x 3 DW 卷积直接连接了 1 x 1 的 PW 卷积而是通过 BN 和 ReLU 层后才连接上 PW 卷积。BN 层作用是对输入层进行归一化，这样可以大大提升模型泛化能力，

##### NB层
$$ \mu = \frac{1}{m} \sum_{i=1}^m x_i$$
$$ \sigma = \frac{1}{m} \sum_{i=1}^m (x_i - \mu)^2$$
$$ \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y_i = \gamma \hat{x_i} + \beta $$
计算 batch 平均值，再计算方差，计算归一化值，引入两个超参数将缩放和平移

##### 降采样方式
- 在 mobileNet 直接采用卷积步长 stride 完成降采样
$$\frac{n + 2p -f}{s} + 1 \times \frac{n + 2p - f}{s} + 1$$  
##### 尺度维度变化
- 输入信息: 224 x 224
- 输出信息: 1 x 1 x 1024
#### 深度可分离卷积(Depthwise Separabable )
##### 标准卷积

![convolutional_filter.jpg](https://upload-images.jianshu.io/upload_images/8207483-fa1674adcd8cd6b2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- 输入(F) $D_F \times D_F \times M$
- 卷积核(K) $D_K \times D_K \times M \times N$
- 输出(F) $D_F \times D_F \times N$

我们从空间和通道来理解卷积过程，每一次卷积在空间上是稀疏连接，通道之间是密集连接。

##### 深度可分离卷积
![DW_PW.jpeg](https://upload-images.jianshu.io/upload_images/8207483-9d6680dca178f20b.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- DW 卷积: 通道数为 1 的卷积核，负责 1 个通道也就是卷积核个数与输入通道输出一致，第一步是探索空间
- PW 卷积: 逐点卷积核大小为 1 x 1 x M ，每一个像素的区域。通道之间丰富联系，所以还需要 PW 卷积来实现通道间的丰富联系。

在 DW 卷积在空间上区域与区域之间的稀疏连接，在通道上是，PW 是特殊普通卷积，特殊在卷积核的大小，每一次卷积空间是一个像素点，通道上也是密集连接。

##### 标准卷积和深度可分离卷积
|   |  标准卷积 | 深度可分离卷积  |
|---|---|---|
| 运算特点  | 每一个卷积核的通道与输入通道相同，每个通道单独做卷积运算后相加   | DW卷积: 一个卷积核只有一个通道，负责一个通道。PW卷积:会将上一步的特征图在通道方向进行扩展  |
|公式表示|$G_{k,l,n} = \sum_{i,j,m} K_{i,j,m,n} F_{k+i-1,l+j-1,m}$|$G_{k,l,m} = \sum_{i,j} K_{i,j,m} F_{k+i-1,l+j-1,m}$|
|算力|$D_KD_KMND_FD_F$|$D_KD_KMD_FD_F + MND_FD_F$|

$G_{k,l,n} = \sum_{i,j,m} K_{i,j,m,n} F_{k+i-1,l+j-1,m}$
- K 表示卷积核
- F 输出
- i,j 表示位置，m 表示通道数，n 表示卷积核个数

$$\frac{D_KD_KMD_FD_F + MND_FD_F}{D_KD_KMND_FD_F} = \frac{1}{N} + \frac{1}{D_K^2}$$

$$\frac{D_KD_KMD_FD_F}{D_KD_KMND_FD_F} + \frac{MND_FD_F}{D_KD_KMND_FD_F} = \frac{1}{N} + \frac{1}{D_K^2}$$

#### 超参数
MobileNet 通过设置两个超参数，实现准确率和延时性之间的平衡。
##### 宽度超参数(Width Multipilier)
引入宽度超参数$\alpha$ 统一规范每层的特征输入和输出的维度 $\alpha \in [0,1]$ 通常设置为1，0.75，0.5，0.25，如果取 1 就是基础的 mobileNet 参数
例如输入通道数$\alpha M$，输出通道数为$\alpha N$，则$D_KD_K \alpha MD_FD_F + \alpha M \alpha ND_FD_F$ 因此对运算消耗减少为原来 mobileNet 的参数 $\alpha^2$倍
##### 分辨率超参数(Resolution Multipilier)
引入宽度超参数$\rho$ 统一规范每层的特征输入和输出的维度 $\rho \in [0,1]$ 。根据输入图像分辨率间接得到的。

例如输入通道数$\rho M$，输出通道数为$\rho N$，则 D_K  D_KM $\rho D_F \rho D_F +  M N \rho D_F\rho D_F$ 因此对运算消耗减少为原来 mobileNet 的参数 $\rho^2$倍


#### 后续版本优化
google 基于之前 mobileNet 版本中问题，通过吸收其他网络结构的优点对现有版本进行优化相继推出 v2 v3 版本。

##### mobileNet V2(Inverted Residuals and Linear Bottlenecks)
在 2018 年 googleNet 提出了 v2 版本的 mobileNet
- 线性瓶颈(Linear Bottlenecks) 在高维空间上，如 ReLU 这种激活函数能有效增加特征的非线性表达，但是仅限于高维空间中，如果降低维度，再使用 ReLU 则会破坏特征，因此在 mobileNets V2 中提出了 Linear Bottlenecks 结构，也就是在执行了降维的卷积层后面，不再加入类似 ReLU 等的激活函数进行非线性转化，这样做的目的是尽可能的不造成信息的丢失。

- 逆残差结构(Inverted residual) 在 ResNet 为了构建更深的网络，提出了 ResNet 的另一种形式，bottleneck 一个 bottleneck 由一个 1 x 1 卷积(降维)，3 x 3 卷积和 1 x 1 卷积(升维)构成。在 MobileNet 中，DW 卷积的层数是输入通道数，本身就比较少，如果跟残差网络中 bottleneck 一样，先压缩，后卷积提取，可得到特征就太少了。采取了一种逆向的方法—先升维，卷积，再降维
![mobilenet_v2_001.jpeg](https://upload-images.jianshu.io/upload_images/8207483-0442fd1e076c3cb1.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


大致看一下 v2 网络结构
![mobilenet_v2_003.jpeg](https://upload-images.jianshu.io/upload_images/8207483-efa5e08c7169fe90.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![mobilenet_v2_002.jpeg](https://upload-images.jianshu.io/upload_images/8207483-70b668a7de0fafa2.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这里还需要说 ReLU6,与不同 ReLU 不同，限制了最大输出为 6 ，这是为了在移动设备也能有很好的数值分辨率。

##### mobileNet V3

在 2019 年 google 提出了 v3 版本，
- 新的激活函数(h-swish): h-swish 是基于 swish 激活函数的改进，所以先了解一下 swish， swish 具备无上届有下界、平滑、非单调的特性。并且 swish 在深层模型上的效果优于 ReLU
$$\begin{aligned}
    swish activation =  f(x) = x sigmoid(\beta x) \\
    h-swish = x \frac{ReLU6(x+3)}{6}
\end{aligned}$$

- 引入SENet 引入轻量级注意力机制网络，通过压缩、激励、给不同层权重。

首次提取是在 2017 年，然后在 2019 年提出最新的版本，是有一位于北京 Monenta 和 Oxford 提出的。这个网络结构设计目的就是让神经网络使用全局信息来增强有用的信息，同时抑制无用的信息。
<img src="../images/mobilenet_v3_002.png">
这里 SE 表示 Squeeze-and-Excitation 也就是分为两个阶段
  - Squeeze 阶段
    $$z\in \mathbb{R}^C \, z_i = F_{squeeze}(u_i) = \frac{1}{H_2 W_2} \sum_x \sum_y u_i(x,y)$$
    在第一阶段(Squeeze) 阶段，对输入进行一个全局平均池化
  - Excitation 阶段
    $$s = F_{ex}(z,W) = \sigma_{sigmoid}(W_2 \sigma_{ReLU}(W_1z)) \, W_1 \in \mathbb{R}^{\frac{C}{r},C}, W_2 \in \mathbb{R}^{C,\frac{C}{r}}$$
    首先经过全局平均池化输出进行两个全连接，这两个全连接层分别使用的是 relu 和 sigmoid激活函数。因为第二层全连接是 sigmoid 所以输出 0 到 1 值，通过 sigmoid 将有用的信息保留，抛弃没有用的信息。

    $$\hat{X_c} = S_cu_c$$
    然后将输出和最初输入作为点乘进行输出
![seblock.png](https://upload-images.jianshu.io/upload_images/8207483-252bbefd0bab1bf1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

    - 调整 Reduction 比率
    根据实验得出 r= 8，16 表现都不错，推荐 16
    - 然后就是在 Squeeze 阶段是使用 Max pooling 还是 Avg pooling，结果是 Avg pooling 更好
 
    - Excitation 的探索
    <img src="../images/seneg_005.jpeg">
    在早期的卷积层提取一些基础共性特征，后期的卷积层偏于提取一些独特的特征，5-2 是拐点(饱和状态)，移除后期的卷积层可以减少参数量，同时模型不会受到太大影响。
##### 网络结构
MobileNet v2 模型中反转残差结构和变量利用 1 x 1 卷积，以便于扩展到高纬的特征空间，虽然对于提取丰富特征进行预测十分重要，但却额外增加计算的开销与延时。为了在保留高纬特征的前提下减少延时，将平均池化前的层移除并用 1 x 1 卷积来计算特征图

#### 简单总结
##### V1
- 深度可分离卷积: DW 和 PW 组合替换常规卷积来达到减少参数数量的目的
- 超参数: 改变输入输出通道数和特征尺寸
##### V2
- 线性瓶颈结构
- 逆向残差结构
##### V3
- h-swish 激活函数
- SENet 结构

