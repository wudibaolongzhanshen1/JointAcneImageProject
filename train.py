import dataset_processing

from resnet50 import *
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from transforms.affine_transforms import RandomRotate
from utils import genLD

device = torch.device("cuda")
print(device)
save_path = "models/ResNet.pth"
batch_size = 32
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RandomRotate(rotation_range=20),
    transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                         std=[0.2814769, 0.226306, 0.20132513])
])
validate_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                         std=[0.2814769, 0.226306, 0.20132513])
])
train_dataset = dataset_processing.DatasetProcessing('dataset/Classification/NNEW_trainval_0.txt',
                                                     'dataset/Classification/JPEGImages', train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,
                          collate_fn=dataset_processing.DatasetProcessing.collate_fn)
train_num = len(train_dataset)
validate_dataset = dataset_processing.DatasetProcessing('dataset/Classification/NNEW_test_0.txt',
                                                        'dataset/Classification/JPEGImages', validate_transform)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                             collate_fn=dataset_processing.DatasetProcessing.collate_fn)
validate_num = len(validate_dataset)
net = ResNet.generate().to(device)
crossEntropyLoss = nn.CrossEntropyLoss().to(device)
klLoss1 = nn.KLDivLoss().to(device)
klLoss2 = nn.KLDivLoss().to(device)
klLoss3 = nn.KLDivLoss().to(device)
optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
best_acc = 0.0


net.eval()
grade_right_num = 0
lesions_right_num = 0
for step, (images, grade_labels, lesions_nums) in enumerate(validate_loader):
    images = images.to(device)
    grade_labels = grade_labels.numpy()
    lesions_nums = lesions_nums.numpy()
    cls, cnt, cnt2cls = net(images)
    grade_right_num += (torch.argmax(cls, 1) == torch.from_numpy(grade_labels).to(device)).sum().item()
    lesions_right_num += (torch.argmax(cnt2cls, 1) == torch.from_numpy(grade_labels).to(device)).sum().item()
acc = (grade_right_num) / (validate_num)
print(f"第0轮acc为:{acc}")


lambda_ = 0.6

for epoch in range(1000):
    net.train()
    running_loss = 0.0
    for step,(images, grade_labels, lesions_nums) in enumerate(train_loader):
        images = images.to(device)
        grade_labels = grade_labels.numpy()
        lesions_nums = lesions_nums.numpy()
        cls, cnt, cnt2cls = net(images)
        ld_cls = torch.from_numpy(genLD.genLD(grade_labels, 3, 4)).to(device)
        ld_cnt = torch.from_numpy(genLD.genLD(lesions_nums, 3, 65)).to(device)
        ld_cnt2cls = torch.from_numpy(genLD.genLD(grade_labels, 3, 4)).to(device)
        loss_cls = klLoss1(torch.log(cls), ld_cls)
        loss_cnt = klLoss2(torch.log(cnt), ld_cnt)
        loss_cnt2cls = klLoss3(torch.log(cnt2cls), ld_cnt2cls)
        loss = (1 - lambda_) * loss_cnt + lambda_ / 2 * (loss_cls + loss_cnt2cls)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss += loss
    print(f"第{epoch + 1}轮running_loss为:{running_loss / len(train_loader)}")

    net.eval()
    grade_right_num = 0
    lesions_right_num = 0
    acc = 0
    for step,(images, grade_labels, lesions_nums) in enumerate(validate_loader):
        images = images.to(device)
        grade_labels = grade_labels.numpy()
        lesions_nums = lesions_nums.numpy()
        cls, cnt, cnt2cls = net(images)
        grade_right_num += (torch.argmax(cls, 1) == torch.from_numpy(grade_labels).to(device)).sum().item()
        lesions_right_num += (torch.argmax(cnt2cls, 1) == torch.from_numpy(grade_labels).to(device)).sum().item()
    acc = (grade_right_num) / (validate_num)
    print(f"第{epoch + 1}轮acc为:{acc}")
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), save_path)
        print(f"第{epoch + 1}轮模型保存成功")



