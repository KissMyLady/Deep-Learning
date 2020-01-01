深度学习Tensorflow安装
====

## Tensorflow可以直接安装  
```Python
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 神经网络需要强大的计算能力  
> 计算能力的提升是第三次人工智能复兴的一个重要因素。实际上，目前深度学习的基础理论在 980年代就已经被提出，但直到2012年基于2块GTX580 GPU训练的 AlexNet发布后，深度学习的真正潜力才得以发挥。  
> 
> 传统的机器学习算法并不像神经网络这样对数据量和计算能力有严苛的要求，通常在 CPU 上串行训练即可得到满意结果。    
> 
> 但是深度学习非常依赖并行加速计算设备，目前的大部分神经网络均使用 NVIDIA GP 和 Google TPU或其他神经网络并行加速芯片训练模型参数。  
> 
> 如围棋程序 AlphaGo Zero 在 64 块 GPU 上从零开始训练了 40 天才得以超越所有的 AlphaGo历史版本；自动网络结构搜索算法使用了800 块 GPU 同时训练才能优化出较好的网络结构  

现在我们开始安装Tensorflow加强包  
### 1. 下载`cuda_10.0.130_411.31_win10.exe`[NVIDIA官网下载](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
点开官网后, 这里可能需要科学上网才能下.  看到链接后, 如果直接点击下载会非常慢  
#### 下载加速方法:   
1. 右击下载链接    
2. 复制这个下载地址   
3. 打开迅雷     
4. 下载  

#### 下载说明:  
这里我们使用 
> 1. CUDA 10.0版本，
> 2. 依次选择 Windows平台，x86_64架构，10系统，
> 3. exe(local)本地安装包，
> 4. 再选择 Download即可下载

#### CUDA安装说明:   
下载完成后，打开安装软件  
> 1. 选择”Custom”选项，点击 NEXT按钮进入      
> 2. 安装程序选择列表，在这里选择需要安装和取消不需要安装的程序。在 CUDA 节点下，取消”Visual Studio Integration”一项    
> 3. 在“Driver components”节点下，比对目前计算机已经安装的显卡驱动“Display Driver”的版本号“Current Version”和 CUDA 自带的显卡驱动版本号“New Version”，   
>> 如果“Current Version”大于“New Version”，则需要取消“Display Driver”的勾，如果小于或等于，则默认勾选即可。
> 4. 设置完成后即可正常安装完成。  

测试是否成功安装:  
1. wind + R,  cmd打开终端
2. 输入`nvcc -V`  
![deep-1](https://github.com/KissMyLady/Deep-Learning/blob/master/Image/deep-1.jpg)  

### 2. 下载cuda加强包`cudnn-10.0-windows10-x64-v7.5.0.56.zip`[cuDNN下载网址](https://developer.nvidia.com/cudnn)  
注意:  
1. 下载的版本号要和你上面下载的CUDA版本号一致   
2. 前面是cudnn自己的版本号, 后面是校验CUDA的版本好, 别搞混了  

#### 安装cuda加强包cudnn:    
1. 下载完成后，解压      
2. 将名为“cuda”的文件夹重命名为“cudnn765”，并复制此文件夹   
3. 进入 CUDA 的安装路径`C:\Program Files\NVIDIA GPUComputing Toolkit\CUDA\v10.0`  
4. 粘贴“cudnn765”文件夹即可，此处会弹出需要管理员权限的对话框，选择继续     


### 3. 配置环境变量   
cudnn文件夹的复制即已完成 cuDNN 的安装，但为了让系统能够感知到 cuDNN 文件的位置，我们需要额外配置 Path 环境变量。
配置方法:   
1. 开文件电脑，在“我的电脑”上右击，选择“属性”，选择“高级系统属性”，选择“环境变量”    
2. 在“系统变量”一栏中选中“Path”环境变量，选择“编辑”     
3. 选择“新建”    
4. 输入我们cuDNN的安装路径`C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v10.0\cudnn765\bin`  
5. 并通过“向上移动”按钮将这一项上移, 将cudnn环境变量置顶   
   
### 4. 最后你应该看到   
1. 环境变量中应该包含     
> `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v10.0\bin`
> `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v10.0\libnvvp`
> `C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v10.0\cudnn765\bin`
下系统变量-环境变量中, 这三个变量一定要有   


### 5. 下载TensorFlow  
`pip install tensorflow-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple`  
意义:  下载`tensorflow-gpu`, 后面的GPU, 就是我们的目的, 可以加强计算能力        
![deep-3](https://github.com/KissMyLady/Deep-Learning/blob/master/Image/deep-3.jpg)  

测试是否成功安装:  
1. 在 cmd命令行输入`ipython`或者`python`进入交互式终端  
2. 输入“import tensorflow as tf”命令，如果没有错误产生，继续输入
3. `tf.test.is_gpu_available()`测试 GPU 是否可用，此命令会打印出一系列以“I”开头的信息(Information)，其中包含了可用的 GPU显卡设备信息    
4. 最后会返回“True”或者“False”，代表了 GPU 设备是否可用   
5. 如果为 True，则 TensorFlow GPU版本安装成功；
6. 如果为 False则安装失败，需要再次检测 CUDA，cuDNN，环境变量等步骤，或者复制错误，从搜索引擎中寻求帮助   

![deep-4](https://github.com/KissMyLady/Deep-Learning/blob/master/Image/deep-4.jpg)  



## 深度学习主流框架    
1.  Theano 是最早的深度学习框架之一，由 Yoshua Bengio 和 Ian Goodfellow 等人开发    
是一个基于 Python 语言、定位底层运算的计算库，Theano 同时支持 GPU 和 CPU 运  
算。由于 Theano 开发效率较低，模型编译时间较长，同时开发人员转投 TensorFlow    
等原因，Theano 目前已经停止维护。  
  
2. Scikit-learn 是一个完整的面向机器学习算法的计算库，内建了常见的传统机器学习算   
法支持，文档和案例也较为丰富，但是 Scikit-learn 并不是专门面向神经网络而设计  
的，不支持 GPU 加速，对神经网络相关层实现也较欠缺。  

3. Caffe 由华人博士贾扬清在 2013 年开发，主要面向使用卷积神经网络的应用场合，并  
不适合其他类型的神经网络的应用。Caffe 的主要开发语言是 C++，也提供 Python 语  
言等接口，支持 GPU 和 CPU。由于开发时间较早，在业界的知名度较高，2017 年  
Facebook 推出了 Caffe 的升级版本 Cafffe2，Caffe2 目前已经融入到 PyTorch 库中。  
  
4. Torch 是一个非常优秀的科学计算库，基于较冷门的编程语言 Lua 开发。Torch 灵活性 
较高，容易实现自定义网络层，这也是 PyTorch 继承获得的优良基因。但是由于 Lua   
语言使用人群较小，Torch 一直未能获得主流应用。  

4. MXNET 由华人博士陈天奇和李沐等人开发，已经是亚马逊公司的官方深度学习框  
架。采用了命令式编程和符号式编程混合方式，灵活性高，运行速度快，文档和案例  
也较为丰富。  

5. PyTorch 是 Facebook 基于原有的 Torch 框架推出的采用 Python 作为主要开发语言的深 
度学习框架。PyTorch 借鉴了 Chainer 的设计风格，采用命令式编程，使得搭建网络和  
调试网络非常方便。尽管 PyTorch 在 2017 年才发布，但是由于精良紧凑的接口设计，  
PyTorch 在学术界获得了广泛好评。在 PyTorch 1.0 版本后，原来的 PyTorch 与 Caffe2  
进行了合并，弥补了 PyTorch 在工业部署方面的不足。总的来说，PyTorch 是一个非常  
优秀的深度学习框架。   

6. Keras 是一个基于 Theano 和 TensorFlow 等框架提供的底层运算而实现的高层框架， 
提供了大量方便快速训练，测试的高层接口，对于常见应用来说，使用 Keras 开发效  
率非常高。但是由于没有底层实现，需要对底层框架进行抽象，运行效率不高，灵活  
性一般。
7. TensorFlow 是 Google 于 2015 年发布的深度学习框架，最初版本只支持符号式编程。  
得益于发布时间较早，以及 Google 在深度学习领域的影响力，TensorFlow 很快成为最  
流行的深度学习框架。但是由于 TensorFlow 接口设计频繁变动，功能设计重复冗余，  
符号式编程开发和调试非常困难等问题，TensorFlow 1.x 版本一度被业界诟病。2019  
年，Google 推出 TensorFlow 2 正式版本，将以动态图优先模式运行，从而能够避免  
TensorFlow 1.x 版本的诸多缺陷，已获得业界的广泛认可。  

#### 目前来看，TensorFlow 和 PyTorch 框架是业界使用最为广泛的两个深度学习框架, TensorFlow 在工业界拥有完备的解决方案和用户基础，PyTorch 得益于其精简灵活的接口设计，可以快速设计调试网络模型，在学术界获得好评如潮。TensorFlow 2 发布后，弥补了 TensorFlow 在上手难度方面的不足，使得用户可以既能轻松上手 TensorFlow 框架，又能无缝部署网络模型至工业系统。本书以 TensorFlow 2.0 版本作为主要框架，实战各种深度学习算法    
