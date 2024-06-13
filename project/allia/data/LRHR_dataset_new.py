from io import BytesIO
# from logging import raiseExceptions
import lmdb
import pydicom  # 引入pydicom库来处理DICOM文件
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import imageio.v2 as imageio  # 保留以处理非DICOM图像
import torch
import numpy as np
import blobfile as bf
import cv2
import os

def resize_and_convert(img, size, resample=cv2.INTER_CUBIC):
    if (img.shape[0] != size):
        img = cv2.resize(img, (size, size), resample)
        # img = cv2.center_crop(img, size)
    return img


def _list_image_files_recursively(data_dir):
    results = []
    try:
        for entry in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, entry)
            if os.path.isfile(full_path):
                ext = os.path.splitext(entry)[-1].lower()
                # print("ext= ",ext)
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".dcm"]:
                    results.append(full_path)
                    # print("ful_path=   ",full_path)

            elif os.path.isdir(full_path):
                results.extend(_list_image_files_recursively(full_path))
    except OSError as e:
        print(f"Error listing files in {data_dir}: {e}")
    return results  # 总是返回一个列表，即使它是空的
    # for entry in sorted(bf.listdir(data_dir)):
    #     full_path = bf.join(data_dir, entry)
    #     ext = entry.split(".")[-1].lower()  # 获取文件扩展名并转换为小写
    #     if bf.isfile(full_path) and ext in ["jpg", "jpeg", "png", "gif", "dcm"]:  # 添加对dcm的支持
    #         results.append(full_path)
    #     elif bf.isdir(full_path):
    #         results.extend(_list_image_files_recursively(full_path))
    # return results



class LRHRDatasetCT(Dataset):
    def __init__(self, dataroot, datatype, exclude_patients=[], include_patients=[],
                 l_resolution=16, r_resolution=128, patch_n=1,
                 split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.patch_n = patch_n
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        all_files = _list_image_files_recursively(dataroot)
        print(len(all_files))

        if self.split == 'train':
            for f in all_files:
                for p in exclude_patients:
                    if p in f:
                        all_files.remove(f)
                        break

            print(len(all_files))
            self.img_paths = all_files

        elif self.split == 'val':
            val_files = []
            for f in all_files:
                for p in include_patients:
                    if p in f:
                        val_files.append(f)
            print(len(val_files))
            self.img_paths = val_files

        self.dataset_len = len(self.img_paths)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        min_max = (-1, 1)

        # 检查文件扩展名，以确定是否使用pydicom或imageio读取
        ext = self.img_paths[index].split(".")[-1].lower()
        if ext == 'dcm':
            dcm = pydicom.dcmread(self.img_paths[index])
            img_HR = dcm.pixel_array.astype(np.float32)  # 获取像素数组
            # 根据需要调整窗宽窗位（这里只是一个示例，你可能需要根据实际的窗宽窗位来调整）
            window_width = dcm.WindowWidth[0]
            window_center = dcm.WindowCenter[0]
            # print("window_width=",window_width," window_center=",window_center)

            img_HR = (img_HR - window_center) / window_width
            img_HR = np.clip(img_HR, -1, 1)  # 确保值在-1到1之间
        else:
            img_HR = imageio.imread(self.img_paths[index]) / 255.  # 对于非DICOM图像，假设像素值在0-255之间

        img_HR = resize_and_convert(img_HR, self.r_res)

        if self.need_LR:
            img_LR = resize_and_convert(img_HR, self.l_res)
            img_SR = resize_and_convert(img_LR, self.r_res // self.patch_n)

            # 将图像转换为torch.FloatTensor并归一化
        if self.need_LR:
            [img_LR, img_SR, img_HR] = [torch.FloatTensor(img).unsqueeze_(0) for img in [img_LR, img_SR, img_HR]]
            [img_LR, img_SR, img_HR] = [img * (min_max[1] - min_max[0]) + min_max[0] for img in
                                        [img_LR, img_SR, img_HR]]
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            img_HR = torch.FloatTensor(img_HR).unsqueeze_(0)
            img_HR = img_HR * (min_max[1] - min_max[0]) + min_max[0]
            return {'HR': img_HR, 'Index': index}


class NoisyCleanDatasetCT(Dataset):
    def __init__(self, dataroot, dataroot1, datatype, exclude_patients=[], include_patients=[],
                 l_resolution=16, r_resolution=128, patch_n=1,
                 split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.patch_n = patch_n
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        all_files = _list_image_files_recursively(dataroot)
        all_files_n = _list_image_files_recursively(dataroot1)
        print(len(all_files))

        if self.split == 'train':
            for f, fn in zip(all_files, all_files_n):
                for p in exclude_patients:
                    if p in f:
                        all_files.remove(f)
                        all_files_n.remove(fn)
                        break

            print(len(all_files))
            self.img_paths = all_files
            self.img_paths_n = all_files_n

        elif self.split == 'val':
            val_files = []
            val_files_n = []
            for f, fn in zip(all_files, all_files_n):
                for p in include_patients:
                    if p in f:
                        val_files.append(f)
                        val_files_n.append(fn)
            print(len(val_files))
            self.img_paths = val_files
            self.img_paths_n = val_files_n

        assert len(self.img_paths) == len(self.img_paths_n)
        self.dataset_len = len(self.img_paths)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        min_max = (-1, 1)

        # 检查文件扩展名，以确定是否使用pydicom或imageio读取
        ext_hr = self.img_paths[index].split(".")[-1].lower()
        ext_lr = self.img_paths_n[index].split(".")[-1].lower()

        if ext_hr == 'dcm':
            dcm_hr = pydicom.dcmread(self.img_paths[index])
            img_HR = dcm_hr.pixel_array.astype(np.float32)  # 获取HR图像的像素数组
        else:
            img_HR = imageio.imread(self.img_paths[index]) / 65535.

        if ext_lr == 'dcm':
            dcm_lr = pydicom.dcmread(self.img_paths_n[index])
            img_LR = dcm_lr.pixel_array.astype(np.float32)  # 获取LR图像的像素数组
        else:
            img_LR = imageio.imread(self.img_paths_n[index]) / 65535.
        img_HR = resize_and_convert(img_HR, self.r_res)

        # img_SR = imageio.imread(self.sr_path[index]) / 65535.
        if self.need_LR:
            img_LR = resize_and_convert(imageio.imread(self.img_paths_n[index]) / 65535., self.l_res)
            # img_SR = resize_and_convert(resize_and_convert(img_HR, self.l_res), self.r_res // self.patch_n)
            img_SR = resize_and_convert(img_LR, self.r_res // self.patch_n)
            # img_SR = imageio.imread(self.lr_path[index]) / 65535.

        if self.need_LR:
            [img_LR, img_SR, img_HR] = [torch.FloatTensor(img).unsqueeze_(0) for img in [img_LR, img_SR, img_HR]]
            [img_LR, img_SR, img_HR] = [img * (min_max[1] - min_max[0]) + min_max[0] for img in
                                        [img_LR, img_SR, img_HR]]
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            # [img_SR, img_HR] = [torch.FloatTensor(img).unsqueeze_(0) for img in [img_SR, img_HR]]
            # [img_SR, img_HR] = [img * (min_max[1] - min_max[0]) + min_max[0] for img in [img_SR, img_HR]]
            # return {'HR': img_HR, 'SR': img_SR, 'Index': index}
            img_HR = torch.FloatTensor(img_HR).unsqueeze_(0)
            img_HR = img_HR * (min_max[1] - min_max[0]) + min_max[0]
            return {'HR': img_HR, 'Index': index}