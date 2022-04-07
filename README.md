[参考的教程链接](https://github.com/udacity/P1_Facial_Keypoints) </br>
建议先看一下参考链接中的东西，我只是将网络模型和训练函数进行了修改，并将其放在GPU上运行了。数据的详细介绍和处理都是教程里面的

## 环境安装
用的工具是`vscode`以及`conda`，`Linux`系统（`Windows`也可以）

- conda创建环境（带ipykernel的）
`conda create -n 环境名称 python=3.8 ipykernel`

- 安装pytorch（先激活刚才创建的环境并进入）（cuda版本要是你的显卡支持的版本才行，具体可以百度查看，我下面的命令是cuda11.1）
`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

如果上面安装报证书错误(There was a problem confirming the ssl certificate: HTTPSConnectionPool(host=‘**.org‘,port=443))用下面的命令：
`pip install --trusted-host download.pytorch.org torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

- 安装相应的cuda或者cudnn
[参考链接](https://blog.csdn.net/qq_44961869/article/details/115954258?utm_source=wechat_session&utm_medium=social&utm_oi=1189675894455939072)

- 安装requirements.txt中的库(先把这个github项目下载到本地然后进入该文件夹中执行)
`pip install -r requirements.txt`

## 数据说明
训练数据和测试数据都放在了`data`文件夹中，关键点信息是在`csv`文件中，图片是在相应的文件夹中，关于数据处理部分可以参考原链接或者`load_data.py`文件

## 网络结构
训练好的模型参数保存在`saved_models`文件夹下，可以直接用`model.py`中的网络来加载训练好的参数看一下效果，如何加载看一下`run.py`文件最后部分
（文件太大没上传上，可以在这个链接中找到链接：https://pan.baidu.com/s/1zqxc7yWc8NmaswqTp5O9rA 
提取码：j49l）

## 训练与测试
文件`run.ipynb`
