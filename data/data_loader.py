#!/user/bin/python
# -*- encoding: utf-8 -*-

from torch.utils import data
import os
from os.path import join, basename
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
import cv2
from .transform import *

def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


class MyDataLoader(data.Dataset):
    """
    Dataloader
    """
    def __init__(self, root='./Data/NYUD', split='train', transform=True):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

        # pre-processing
        if self.transform:
            self.trans = Compose([
                ColorJitter(
                    brightness = 0.5,
                    contrast = 0.5,
                    saturation = 0.5),
                #RandomCrop((512, 512))
                ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()

            label = Image.open(join(self.root, lb_file))
            img = Image.open(join(self.root, img_file))

            if self.transform:
                im_lb = dict(im = img, lb = label)
                im_lb = self.trans(im_lb)
                img, label = im_lb['im'], im_lb['lb']

            img = np.array(img, dtype=np.float32)
            img = prepare_image_PIL(img)

            label = np.array(label, dtype=np.float32)

            if label.ndim == 3:
                label = np.squeeze(label[:, :, 0])
            assert label.ndim == 2

            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label>0, label<=73.5)] = 2
            label[label > 73.5] = 1

            return img, label, basename(img_file).split('.')[0]
        else:
            img_file = self.filelist[index].rstrip()
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, basename(img_file).split('.')[0]


class NYUD_Loader(data.Dataset):
    def __init__(self, root='data/', split='train', transform=False, threshold=0.4, setting=['image']):
        self.root = root
        self.split = split
        self.threshold = 128
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                self.root, '%s-train.lst' % (setting[0]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                self.root, '%s-test.lst' % (setting[0]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        scale = 1.0
        if self.split == "train":
            img_file, lb_file, scale = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            scale = float(scale.strip())
            pil_image = Image.open(os.path.join(self.root, lb_file))
            if scale < 0.99:  # which means it < 1.0
                W = int(scale * pil_image.width)
                H = int(scale * pil_image.height)
                pil_image = pil_image.resize((W, H))
            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            if scale < 0.9:
                img = img.resize((W, H))
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class BSDS_VOCLoader(data.Dataset):
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb, basename(img_file).split('.')[0]
        else:
            img_name = Path(img_file).stem
            return img, img_name


class Multicue_Loader(data.Dataset):


    def __init__(self, root='data/', split='train', transform=False, threshold=0.3, setting=['boundary', '1']):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                self.root, 'train_pair_%s_set_%s.lst' % (setting[0], setting[1]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                self.root, 'test_%s_set_%s.lst' % (setting[0], setting[1]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name