import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import dataset_processing
from transforms.affine_transforms import RandomRotate

device = torch.device("cpu")
save_path = "models/ResNet.pth"
batch_size = 2
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
validate_dataset = dataset_processing.DatasetProcessing('dataset/Classification/NNEW_test_0.txt',
                                                        'dataset/Classification/JPEGImages', validate_transform)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,
                             collate_fn=dataset_processing.DatasetProcessing.collate_fn)
train_dataset = dataset_processing.DatasetProcessing('dataset/Classification/NNEW_trainval_0.txt',
                                                     'dataset/Classification/JPEGImages', train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=dataset_processing.DatasetProcessing.collate_fn)
train_num = len(train_dataset)
validate_num = len(validate_dataset)
for step,(images, grade_labels, lesions_nums) in enumerate(validate_loader):
    print(f'{step}步:images_shape:{images.shape},grade_labels:{grade_labels},lesions_nums:{lesions_nums}')