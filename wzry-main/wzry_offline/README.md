# 1 代码讲解

## 1.1 项目目录

offline分为两个程序，sampler为数据采集程序，trainer为模型训练程序

## 1.2 sampler
log 目录存放着训练时产生的日志文件。

models 目录存放着推理使用的模型。

experineces目录存放着采集的经验数据文件pth格式

new_wzry_ai 是主要代码文件。
>config 存放配置信息；

>core 目录存放模型核心代码，如模型网络架构，经验池设计；

>environment 目录为环境模块，负责模拟游戏中的动态环境，如奖励函数的实现；

>LLM_Utils 目录存放着大语言模型工具单元；若要启动该模块，请将config目录下的default_config.py 第190行设置为自己的api

>runs 存放训练的 yolo 模型，用来识别敌我双方英雄位置；

>stats 存放着全局的训练信息，如胜率；

>template 存放着 opencv 的模板匹配图片；

>training 存放着训练代码，例如训练的总架构；

>utils 存放着工具模块，例如血量检测、经济检测、技能冷却检测等等。

>test 测试用的代码与图片

采集时，请运行 main.py 文件

## 1.3 trainer


log 目录存放着训练时产生的日志文件。

models 目录存放着训练产生的模型。

experineces目录存放着训练需要的经验数据文件pth格式

runs目录下存放着训练产生的loss图像以及tensorboard的数据文件，可以使用tensorboard查看更详细的训练过程

new_wzry_ai 是主要代码文件。

>config 存放配置信息；

>core 目录存放模型训练代码，如模型网络架构；

>training 存放着训练代码，例如训练的总架构；

>utils 存放着工具模块

训练时，请运行 main.py 文件