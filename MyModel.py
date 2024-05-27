import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class resnet50(nn.Module):
    def __init__(self):
        super(resnet50,self).__init__()
        self.net = models.resnet50(weights=ResNet50_Weights)
        # 置为空即删除全连接层
        self.net.fc = nn.Sequential()
        # fc是分级任务的输出层
        self.fc = nn.Linear(2048, 4)
        # counting是计数任务的输出层
        self.counting = nn.Linear(2048, 65)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.net(x)
        cls = self.fc(x)
        cls = self.softmax(cls)
        cnt = self.counting(x)
        cnt = self.softmax(cnt)
        cnt2cls = torch.stack([cnt[:, :5].sum(1), cnt[:, 5:20].sum(1), cnt[:, 20:50].sum(1), cnt[:, 50:].sum(1)], 1)
        return cls, cnt, cnt2cls



