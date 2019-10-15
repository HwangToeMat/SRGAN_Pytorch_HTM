import os
import glob
import h5py
import cv2
from PIL import Image
import numpy as np

def mod_crop(image, scale = 4):
    if len(image.shape) ==3:
        h = image.shape[0]
        w = image.shape[1]
        h = h - np.mod(h,scale)
        w = w - np.mod(w,scale)
        return image[0:h,0:w,:]
    else:
        h = image.shape[0]
        w = image.shape[1]
        h = h - np.mod(h,scale)
        w = w - np.mod(w,scale)
        return image[0:h,0:w]

def sub_img(input, i_size = 96, stride = 90):
    sub_img = []
    for h in range(0, input.shape[0] - i_size + 1, stride):
        for w in range(0, input.shape[1] - i_size + 1, stride):
            sub_i = input[h:h+i_size,w:w+i_size]
            sub_img.append(sub_i)
    return sub_img

def load_img(file_path):
    dir_path = os.path.join(os.getcwd(), file_path)
    img_path = glob.glob(os.path.join(dir_path, '*.jpg'))
    return img_path

def read_img(img_path):
    # read image
    image = cv2.imread(img_path)
    return image

def img_downsize(img, ds):
    dst_list = []
    img_list = []
    for _ in img:
        img_list.append(_.reshape(3, 96, 96))
        dst = cv2.resize(_, dsize=(0, 0), fx=1/ds, fy=1/ds, interpolation=cv2.INTER_CUBIC)
        dst_list.append(dst.reshape(3, 24, 24))
    return dst_list, img_list

def save_h5(sub_ip, sub_la, savepath = 'data/train.h5'):
    path = os.path.join(os.getcwd(), savepath)
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('input', data=sub_ip)
        hf.create_dataset('label', data=sub_la)

def data_aug(file_path = 'data/Train', savepath = 'data/train.h5'):
    sub_ip = []
    sub_la = []
    num = 1
    img_path = load_img(file_path)
    for _ in img_path:
        image = read_img(_)
        md_image = mod_crop(image, 4)
        sub_image = sub_img(md_image, 96, 90)
        input, label = img_downsize(sub_image, 4)
        sub_ip += input
        sub_la += label
        print('data no.',num)
        num += 1
    sub_ip = np.asarray(sub_ip)
    sub_la = np.asarray(sub_la)
    print('input shape : ',sub_ip.shape)
    print('label shape : ',sub_la.shape)
    save_h5(sub_ip, sub_la, savepath)
    print('---------save---------')

if __name__ == '__main__':
    print('starting data augmentation...')
    data_aug()
