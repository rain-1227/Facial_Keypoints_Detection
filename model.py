import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 36864是计算出来的，或者用网络运行一下前向传播
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=136)

        # Droopouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)

        # Second - Convolution + Activation + Pooling + Dropout
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        #print("Second size: ", x.shape)

        # Third - Convolution + Activation + Pooling + Dropout
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        #print("Third size: ", x.shape)

        # Forth - Convolution + Activation + Pooling + Dropout
        x = self.drop4(self.pool(F.relu(self.conv4(x))))

        # Flattening the layer
        x = x.view(x.size(0), -1)

        x = self.drop5(F.relu(self.fc1(x)))
        x = self.drop6(F.relu(self.fc2(x)))

        x = self.fc3(x)

        return x


def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet152(pretrained=True)

    # 改变输出层
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 136))
    finetune_net.features.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # 模型参数转到cpu或可用的gpu上
    finetune_net = finetune_net.to(devices[0])

    # 冻结参数
    # for param in finetune_net.features.parameters():
    #     param.requires_grad = False

    return finetune_net


