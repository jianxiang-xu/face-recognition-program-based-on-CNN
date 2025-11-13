import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            # 12x12x3
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            # 10x10x10
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            # 5x5x10
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            # 3x3x16
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            # 1x1x32
            nn.PReLU()  # PReLU3
        )
        # detection
        self.conv4_1 = nn.Linear(32,1,bias=False)

    def forward(self, x):
        x = self.pre_layer(x)
        x = torch.flatten(x, start_dim=-3, end_dim=-1)
        det = torch.sigmoid(self.conv4_1(x))
        # 新增：去掉最后一个维度，使输出形状为 [batch_size]
        det = det.squeeze(-1)
        return det


import torch.nn.functional as F

class MLP_Net(nn.Module):
    def __init__(self):
        super(MLP_Net, self).__init__()
        # 输入维度：12x12x3 = 432
        self.fc1 = nn.Linear(12*12*3, 128)  # 第一层：432 → 128
        self.fc2 = nn.Linear(128, 32)       # 第二层：128 → 32
        self.fc3 = nn.Linear(32, 1)# 第三层：32 → 1（直接输出单个值）

    def forward(self, x):
        # 展平输入张量 (batch_size, 432)
        x = x.reshape(-1, 12*12*3)
        
        # 第一层线性变换 + ReLU激活
        x = F.relu(self.fc1(x))
        # 第二层线性变换 + ReLU激活
        x = F.relu(self.fc2(x))
        # 第三层直接输出（无激活函数）
        x = torch.sigmoid(self.fc3(x)).squeeze(-1)
        
        return x
    

class R_Net(nn.Module):
    def __init__(self):
        super(R_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            # 24x24x3
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            # 22x22x28
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            # 10x10x28
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            # 8x8x48
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            # 3x3x48

        )
        # 2x2x64
        self.conv4 = nn.Linear(48 * 3 * 3, 64)  # conv4
        # 128
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        det = torch.sigmoid(self.conv5(x)).squeeze(-1)

        return det
    
class O_Net(nn.Module):
    def __init__(self):
        super(O_Net, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()  # prelu4
        )
        self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        landmark = self.conv6_3(x)
        return landmark


