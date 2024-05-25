import torch
from torch import nn


# BasicBlock：为18和34层的残差结构，Block中所有卷积层的卷积核数都一样
class BasicBlock(nn.Module):
    # expansion用来区分BasicBlock和Bottleneck，Bottleneck的残差结构的最后一个卷积层的输出channel为4*之前几层输出channel，因此其expansion为4
    expansion = 1

    # out_channel为残差结构输出特征矩阵的channel数(也为主分支上卷积层的卷积核数)，
    # 第一种实线残差结构所有的卷积层stride默认为1，第二种虚线残差结构stride不全为1
    # 参数中stride为主分支第一个卷积层以及shortcut中卷积的stride（实线结构中为1，虚线结构中为2）
    # downsample下采样为True时，此残差block的shortcut需要调整channel来与主路径的channel匹配，之后再相加
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()
        # stride（实线结构中为1，虚线结构中为2）
        stride = 1
        if downsample:
            stride = 2
        # 18和34层的残差结构的kernelsize都为3
        # 18，34层实线残差结构stride为1，为保证输出特征矩阵宽高不变，padding也为1
        # 18，34层虚线残差结构主分支第一个卷积层以及shortcut中卷积的stride为2，为保证输出特征矩阵宽高变为1/2，padding也为1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # BatchNorm2d(param),param:输入BN层的特征矩阵的深度/channel数
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        # 这里的stride为1，因为无论实线残差结构还是虚线残差结构的conv2的stride都为1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        # 如果不是第二种虚线残差结构（需要调整shortcut的channel），则shortcut_output就为x
        shortcut_output = x
        if self.downsample is not False:
            # shortcut分支要调整channel
            shortcut_output = self.shortcut_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut_output
        x = self.relu(x)
        return x


# Bottleneck：为50,101,152层的残差结构，Block中最后一个卷积层的卷积核数量为之前两层卷积核数量的4倍
class BottleNeck(nn.Module):
    # expansion用来区分BasicBlock和Bottleneck，Bottleneck的残差结构的最后一个卷积层的输出channel为4*之前几层输出channel，因此其expansion为4
    expansion = 4

    # 第一种实线残差结构所有的卷积层stride默认为1，第二种虚线残差结构stride不全为1
    # 参数中stride为主分支第二个卷积层以及shortcut中卷积的stride（实线结构中为1，虚线结构中为2），注意此处与较少层残差结构BasicBlock有些许不同，可参考残差结构图
    # out_channels为主分支前两个卷积层的卷积核个数
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BottleNeck, self).__init__()
        # stride（实线结构中为1，虚线结构中为2）
        stride = 1
        if downsample:
            stride = 2
        self.relu = nn.ReLU(True)
        # 高层实/虚线残差结构stride均为1，为保证输出特征矩阵宽高不变，padding为0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # BatchNorm2d(param),param:输入BN层的特征矩阵的深度/channel数
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 高层实线残差结构stride为1，为保证输出特征矩阵宽高不变，padding为1  (in-3+2*p+1=in)
        # 高层虚线残差结构stride为2，为保证输出特征矩阵宽高减半，padding为1  [(in-3+2*p)/2+1=in/2]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # 保证输出矩阵宽高不变
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        # 虚线残差结构宽高减半
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        # 如果不是第二种虚线残差结构（需要调整shortcut的channel），则shortcut_output就为x
        shortcut_output = x
        if self.downsample is not False:
            # shortcut分支要调整channel
            shortcut_output = self.shortcut_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += shortcut_output
        x = self.relu(x)
        return x


# 第三种残差结构
class MyBottleNeck(nn.Module):
    expansion = 4

    # out_channels为主分支前两个卷积层的卷积核个数
    def __init__(self, in_channels, out_channels):
        super(MyBottleNeck, self).__init__()
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 保证输出矩阵宽高不变
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        # 虚线残差结构宽高也不变
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shortcut_output = self.shortcut_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += shortcut_output
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    @staticmethod
    def generate():
        block_num_array = [3, 4, 6, 3]
        block = BottleNeck
        return ResNet(block, block_num_array)

    def __init__(self, block, block_num_array):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_x_without_max_pool = self._make_conv2_x_without_max_pool(block, block_num_array[0])
        self.conv3_x = self._make_layer(block, 64 * block.expansion, 128, block_num_array[1])
        self.conv4_x = self._make_layer(block, 128 * block.expansion, 256, block_num_array[2])
        self.conv5_x = self._make_layer(block, 256 * block.expansion, 512, block_num_array[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # fc是分级任务的输出层
        self.fc = nn.Linear(512 * block.expansion, 4)
        # counting是计数任务的输出层
        self.counting = nn.Linear(512 * block.expansion, 65)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x_without_max_pool(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        cls = self.fc(x)
        cls = self.softmax(cls)
        cnt = self.counting(x)
        cnt = self.softmax(cnt)
        cnt2cls = torch.stack([cnt[:, :5].sum(1), cnt[:, 5:20].sum(1), cnt[:, 20:50].sum(1), cnt[:, 50:].sum(1)], 1)
        return cls, cnt, cnt2cls

    # block：传入的类名，BasicBlock或BottleNeck
    # block_num：残差模块的数量
    def _make_conv2_x_without_max_pool(self, block, block_num):
        layers = []
        if block == BasicBlock:
            for _ in range(0, block_num):
                # 实线结构
                layers.append(block(in_channels=64, out_channels=64, downsample=False))
        else:
            # 特殊虚线结构
            layers.append(MyBottleNeck(64, 64))
            for _ in range(1, block_num):
                # 实线结构
                layers.append(block(in_channels=256, out_channels=64, downsample=False))
        return nn.Sequential(*layers)

    # 返回：构造的conv3_x/conv4_x...
    # out_channels：主分支前两个卷积层卷积核的个数
    def _make_layer(self, block, in_channels, out_channels, block_num):
        layers = [block(in_channels=in_channels, out_channels=out_channels, downsample=True)]
        for _ in range(1, block_num):
            layers.append(block(in_channels=out_channels*block.expansion, out_channels=out_channels, downsample=False))
        return nn.Sequential(*layers)
