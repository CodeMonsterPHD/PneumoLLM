import random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import numpy as np
import torch
class PneumDataset(Data.Dataset):
    def __init__(self, args, split):
        super(PneumDataset, self).__init__()
        self.args = args
        self.split = split
        self.image_path = os.path.join(args.data_root, split)
        img_set = os.listdir(self.image_path)
        img_set.sort()
        self.img_set = img_set
        print(f"number of images: {len(self.img_set)}\n")
        # NOTICE: we remove the resize operation, thus, the image should be resized to 224x224 ahead.
        #         transforms.Resize((224, 224), interpolation=Image.BICUBIC)
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def __getitem__(self, idx):
        img_name = self.img_set[idx]
        image_path = os.path.join(self.image_path, img_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        if 'Health' in img_name:
            label = 0
        elif 'Sick' in img_name:
            label = 1
        else:
            assert 'unknown image'

        if self.split == 'train':
            return image, label
        elif self.split == 'val':
            return image, label, img_name

    def __len__(self):
        return len(self.img_set)

    def shuffle_list(self, list):
        random.shuffle(list)

class Shanxi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 seed=0
                 ):
        super(Shanxi_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

        # if self.train:
        #     self.data_aug = transforms.Compose([
        #         # xrv.datasets.ToPILImage(),
        #         # transforms.RandomAffine(45,
        #         #                         translate=(0.15, 0.15),
        #         #                         scale=(1.0 - 0.15, 1.0 + 0.15)),
        #         transforms.ToTensor()
        #     ])

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

        ####### pathology masks ########
        # Get our classes.

        # self.tr = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(DSIZE)])

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return sample

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.data_root = '/disk1/wjr/dataset/shanxi_dataset/'
    cfg=Cfg()
    dataset=Shanxi_Dataset(imgpath=cfg.data_root + 'segimages2_224_resize',
                                  txtpath=cfg.data_root + 'small_val1.txt')
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)

    for sample in data_loader:
        image=sample['img']
        print(image.mean())
