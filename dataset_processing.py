import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DatasetProcessing(Dataset):

    #config_path:txt file path. image_dir_path: image所在文件夹的path
    #根据config_path读取数据
    def __init__(self,config_path:str,image_dir_path:str,transform=None):
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.config_path = config_path
        self.image_dir_path = image_dir_path
        self.transform = transform
        self.__load_data__()

    def __getitem__(self, index):
        # 根据索引返回数据
        img_path = os.path.join(self.image_dir_path, self.image_filenames[index])
        img = Image.open(img_path)
        grade = self.grade_labels[index]
        lesions_num = self.lesions_nums[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, grade, lesions_num


    def __len__(self):
        # 返回数据的长度
        return len(self.image_filenames)

    # 以下可以不在此定义。

    # 如果不是直接传入数据data，这里定义一个加载数据的方法
    def __load_data__(self):
        fp = open(self.config_path, 'r')
        self.image_filenames = []
        self.grade_labels = []
        self.lesions_nums = []
        for line in fp.readlines():
            filename, grade_label, lesions_num = line.split()
            self.image_filenames.append(filename)
            self.grade_labels.append(int(grade_label))
            self.lesions_nums.append(int(lesions_num))
        self.image_filenames = np.array(self.image_filenames)
        self.grade_labels = np.array(self.grade_labels)
        self.lesions_nums = np.array(self.lesions_nums)

    @staticmethod
    def collate_fn(batch):
        # dataset = ((0.56998216, 0.72663738, 0.3706266),(0.3403586 , 0.13931333, 0.71030221)) = (imgarray,label1,label2)。为tuple(tulple,tulple)
        # batch为zip(tuple(tulple,tulple)) = ((img1, label1), (img2, label2), (img3, label3))
        # batch = zip(dataset[0], dataset[1]) = zip:((0.56998216, 0.3403586), (0.72663738, 0.13931333), (0.3706266, 0.71030221))
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # zip()可以理解为压缩，zip(*)可以理解为解压,但是解压后还是zip类型，需要转变为tuple
        # tuple(zip(*batch)) = ((img1, img2), (i1_label1, i2_label1), (i1_label2, i2_label2)...)
        images, grade_labels, lesions_nums = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        grade_labels = torch.as_tensor(grade_labels)
        lesions_nums = torch.as_tensor(lesions_nums)
        return images, grade_labels, lesions_nums
