# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import captcha_setting
import cv2

class mydataset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        
        img = cv2.imread(image_root)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,0,11,2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)
        try:
            label = ohe.encode(image_name.split('_')[0].upper()) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        except IndexError as e:
            print(image_name)
            label = '73NI'
        return img1, label

transform = transforms.Compose([
    transforms.Resize((captcha_setting.IMAGE_HEIGHT,captcha_setting.IMAGE_WIDTH)),
    #transforms.ColorJitter(contrast = (1,2)),
    transforms.Grayscale(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Resize((captcha_setting.IMAGE_HEIGHT,captcha_setting.IMAGE_WIDTH)),
    #transforms.Grayscale(),
    #transforms.ColorJitter(contrast = (1,2)),
    #transforms.RandomRotation(5),
    transforms.ToTensor(),
    
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_data_loader():

    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def get_test_data_loader():
    dataset = mydataset(captcha_setting.TEST_DATASET_PATH, transform=transform_test)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_predict_data_loader():
    dataset = mydataset(captcha_setting.PREDICT_DATASET_PATH, transform=transform_test)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_train_fine_data_loader():

    dataset = mydataset(captcha_setting.FINE_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)