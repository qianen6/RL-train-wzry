# wzry
学校：华东师范大学

指导教授： 李洋

成员： 南昊天 项豪杰 刘成魁 赵元川

特邀成员：夏伟豪

模拟器分辨率改为 1600\*720

### 1.环境配置

#### 1.1 代码环境配置

（推荐使用 anaconda 创建 python 版本为 3.12 的虚拟环境，便于之后资源管理）

本项目使用 pip 作为环境部署工具，在本项目的根目录下输入如下指令。

    pip install -r requirements.txt

这会下载本项目在代码层面上的所有第三方依赖。随后需要下载 PyTorch 和 CUDA，输入如下指令。

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

接下来需要安装用于 GPU 推理的 onnxruntime-gpu。
如果 CUDA 的版本是 11 则输入如下指令进行安装。

    pip install onnxruntime-gpu

如果 CUDA 的版本是 12 则输入如下指令进行安装。

    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

#### 1.2 模拟器环境配置

首先安装模拟器（模拟器推荐使用 mumu 模拟器，安装包放在该项目的 wzry/new_wzry_ai/app_tools/mumu 目录下，可以直接使用），接下来我们需要启动模拟器，并将窗口命名为 wzry_ai，分辨率设置为 1600*720，同时确保屏幕缩放率为 100%。项目启动后程序会自动搜寻名字为 wzry_ai 的窗口并从中获取数据开始训练。

在模拟器中安装王者荣耀程序，并进行键位配置，若使用 mumu 模拟器，可采用该方案（分享《王者荣耀》按键操作方案，复制此分享码：

    mumu0364377751

，打开 MuMu 模拟器，在方案管理中导入使用！方案适用分辨率：1600*720）

若为其他模拟器，可参考下图：
![img.png](assets/img.png)

同时，我们要修改王者荣耀内部布局，总布局为右手模式，局内出装位置为右侧
![img.png](assets/img1.png)

轮盘侧边距与技能侧边距均调整为最低
![img.png](assets/img2.png)

#### 1.3 第三方工具环境配置

##### 1.3.1 OCR 配置

因为项目中使用了 OCR 来识别图片中的相关字符数字，因此，要在本机配置 OCR 工具。

工具已存放在该项目下的 wzry/new_wzry_ai/app_tools/Tesseract-OCR 目录下，需要将该目录的绝对路径添加到系统的环境变量中

##### 1.3.2 ChatGPT 设置

请将 wzry/new_wzry_ai/config/default_config.py 文件中第 180 行变量 openai_api_key_4_0_2 设置为自己的 chatgpt api_key

### 2 代码讲解

#### 2.1 项目目录结构讲解

assets 目录下存储着 Readme 文档下的图片资源。

log 目录存放着训练时产生的日志文件。

Models 目录存放着训练产生的模型。

New_wzry_ai 是主要代码文件。

>其中 app_tools 存放第三方工具：

>config 存放配置信息；

>core 目录存放模型训练代码，如模型网络架构；

>environment 目录为环境模块，负责模拟游戏中的动态环境，如奖励函数的实现；

>LLM_Utils 目录存放着大语言模型工具单元；

>runs 存放训练的 yolo 模型，用来识别敌我双方英雄位置；

>stats 存放着全局的训练信息，如胜率；

>template 存放着 opencv 的模板匹配图片；

>training 存放着训练代码，例如训练的总架构；

>utils 存放着工具模块，例如血量检测、经济检测、技能冷却检测等等。

训练时，请运行 main.py 文件
