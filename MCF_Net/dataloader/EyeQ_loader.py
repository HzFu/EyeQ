# encoding: utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
from sklearn import preprocessing
import pandas as pd


def load_eyeQ_excel(data_dir, list_file, n_class=3):
    image_names = []
    labels = []
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(range(n_class)))
    df_tmp = pd.read_csv(list_file)
    img_num = len(df_tmp)

    for idx in range(img_num):
        image_name = df_tmp["image"][idx]
        image_names.append(os.path.join(data_dir, image_name[:-5] + '.png'))

        label = lb.transform([int(df_tmp["quality"][idx])])
        labels.append(label)

    return image_names, labels


class DatasetGenerator(Dataset):
    def __init__(self, data_dir, list_file, transform1=None, transform2=None, n_class=3, set_name='train'):

        image_names, labels = load_eyeQ_excel(data_dir, list_file, n_class=3)

        self.image_names = image_names
        self.labels = labels
        self.n_class = n_class
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train':
            label = self.labels[index]
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), torch.FloatTensor(label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)

