# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN,resnet

from PIL import Image

import captcha_setting
from torchvision import transforms

def captcha_predict(image):
    
    #接受PIL灰度图像图像输入，返回识别结果
    
    cnn = resnet()
    cnn.eval()
    cnn.load_state_dict(torch.load('./model/model_best_test_loss_num.pkl'))
    print("load cnn net.")
    
    transform = transforms.Compose([
    transforms.Resize((captcha_setting.IMAGE_HEIGHT,captcha_setting.IMAGE_WIDTH)),
    transforms.ToTensor(),
    ])
        
    image = transform(image).unsqueeze(0)
    predict_label = cnn(image)

    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    c = '%s%s%s%s' % (c0, c1, c2, c3)
    print(c)
    
    if '=' in ''.join([c0,c1,c2,c3]):
        
        try:
            test_num1 = int(c0)
            test_num2 = int(c2)
        except ValueError as e:
            return c
        
        if c1 == '!':
            return int(c0)//int(c2) 
        elif c1 == '#':
            return int(c0)*int(c2) 
        elif c1 == '+':
            return int(c0)+int(c2) 
        elif c1 == '-':
            return int(c0)-int(c2) 
        else:
            pass
        
    else:
        pass
    
    return c


if __name__ == '__main__':
    
    img_file = r'C:\Users\MSI-NB\Desktop\dl_captacha\1#5=_9.png'
    img = Image.open(img_file).convert('L')
    result = captcha_predict(img)

