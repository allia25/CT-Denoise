import torch
import model as Model
import argparse
import os
import numpy as np
import yaml
import json
from collections import OrderedDict
import cv2
import pydicom
import metrics


# import imageio.v2 as imageio  # 不再需要imageio库，因为我们不再读取PNG文件

# 读取DCM图片的函数（保留原样）
def read_dcm_img(path):
    """读取DCM图片并返回归一化后的numpy数组"""
    print(f"Reading DCM image from {path}")
    dcm = pydicom.dcmread(path)
    dcm_metas = dcm
    print(f"DCM image loaded with shape {dcm.pixel_array.shape}")
    pixel_array = dcm.pixel_array.astype(np.float32)
    min_val, max_val = pixel_array.min(), pixel_array.max()
    normalized_pixel_array = (pixel_array - pixel_array.min()) / (dcm.pixel_array.max() - dcm.pixel_array.min())
    print(f"DCM image normalized to range [0, 1]")
    return normalized_pixel_array,min_val, max_val,dcm_metas


def save_dcm_img(img, path, original_min, original_max, original_dcm_meta):
    print("Saving DCM image to path:", path)
    # 假设img是[0, 1]范围的numpy数组，需要转换为uint16类型，范围0-65535
    # pixel_array = (img * 65535).astype(np.uint16)
    pixel_array = (img * (original_max - original_min) + original_min).astype(np.uint16)
    pixel_array = pixel_array.clip(0, 65535).astype(np.uint16)

    # 创建一个新的DCM数据集对象并拷贝元信息（除了PixelData）
    dcm = pydicom.Dataset()
    # 复制原始DCM的元信息（除了PixelData）
    dcm = original_dcm_meta.copy()
    # del dcm.PixelData  # 删除原始的PixelData

            # 设置新的PixelData和其他必要的元信息（如果需要）
    dcm.PixelData = pixel_array.tobytes()
    dcm.Rows, dcm.Columns = pixel_array.shape

    # 使用pydicom.dcmwrite来保存DCM文件，它会自动处理文件元数据和传输语法
    pydicom.dcmwrite(path, dcm)

    # # 创建一个新的DCM数据集对象
    # dcm = pydicom.Dataset()
    # dcm.Rows, dcm.Columns = 512, 512  # 设置尺寸为512x512
    # dcm.BitsAllocated = 16
    # dcm.BitsStored = 16
    # dcm.HighBit = 15
    # dcm.PixelRepresentation = 0
    # dcm.PhotometricInterpretation = "MONOCHROME2"
    # dcm.PixelData = pixel_array.tobytes()  # 设置像素数据
    # dcm.is_little_endian = True  # 假设使用小端字节序
    # dcm.is_implicit_VR = True  # 假设使用隐式VR
    # # 使用pydicom.dcmwrite来保存DCM文件，它会自动处理文件元数据和传输语法
    # pydicom.dcmwrite(path, dcm)

# 主程序部分
if __name__ == "__main__":
    # 参数设置部分
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Dn_Liver_128.yaml',
                        help='yaml file for configuration')
    args = parser.parse_args()
    print("Loading configuration file:", args.config)
    cfg = yaml.load(open(args.config), Loader=yaml.FullLoader)

    json_str = ''
    with open(cfg['model']['cfg_path'], 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    print("Loading model configuration from JSON string")
    model_cfg = json.loads(json_str, object_pairs_hook=OrderedDict)

    if len(model_cfg['gpu_ids']) > 1:
        model_cfg['distributed'] = True
    else:
        model_cfg['distributed'] = False

    model_cfg['path']['resume_state'] = cfg['model']['pretrained_path']
    root = cfg['data']['root']
    input_folder = cfg['data']['input_folder']
    cond_folder = cfg['data']['cond_folder'] if cfg['data']['cond_folder'] != 'None' else None
    output_folder = cfg['data']['output_folder']
    target_folder = cfg['data']['target_folder'] if cfg['data']['target_folder'] != 'None' else None
    res = cfg['data']['res']
    length = cfg['data']['len']
    if_ddim = cfg['diffusion']['ddim']
    ddim_eta = cfg['diffusion']['ddim_eta']
    mode = cfg['dn']['mode']
    lam0 = cfg['dn']['lam0']
    a, b, c = cfg['dn']['a'], cfg['dn']['b'], cfg['dn']['c']
    resume = cfg['dn']['resume']
    mean_num = cfg['dn']['mean_num']
    bs = cfg['dn']['bs']

    # 数据处理部分（修改以处理DCM输入）
    print("Processing input files from folder:", os.path.join(root, input_folder))
    low_dcm_files = sorted(os.listdir(os.path.join(root, input_folder)))
    cond_dcm_files = sorted(os.listdir(os.path.join(root, cond_folder))) if cond_folder is not None else None
    target_dcm_files = sorted(os.listdir(os.path.join(root, target_folder))) if target_folder is not None else None
    if length == -1:
        length = len(low_dcm_files)
    print("Creating output folder if it doesn't exist:", os.path.join(root, output_folder))
    os.makedirs(os.path.join(root, output_folder), exist_ok=True)
    inputs, conds, targets = [], [], []
    original_dcm_mins, original_dcm_maxs, original_dcm_metas = [], [], []

    for i, low_name in enumerate(low_dcm_files):
        input_path = os.path.join(root, input_folder, low_name)
        print("Reading input DCM image:", input_path)
        input, original_min, original_max,dcm_meta = read_dcm_img(input_path)  # 使用read_dcm_img函数读取DCM图像
        input = cv2.resize(input, (res, res), cv2.INTER_CUBIC)

        cond = None
        if cond_folder is not None:
            cond_path = os.path.join(root, cond_folder, cond_dcm_files[i])
            if os.path.exists(cond_path):
                print("Reading condition DCM image:", cond_path)
                cond, original_min, original_max,dcm_meta = read_dcm_img(cond_path)  # 使用read_dcm_img函数读取DCM图像
                cond = cv2.resize(cond, (res, res), cv2.INTER_CUBIC)

        target = None
        if target_folder is not None:
            target_path = os.path.join(root, target_folder, target_dcm_files[i])
            if os.path.exists(target_path):
                print("Reading target DCM image:", target_path)
                target, original_min, original_max,dcm_meta = read_dcm_img(target_path)  # 使用read_dcm_img函数读取DCM图像
                target = cv2.resize(target, (res, res), cv2.INTER_CUBIC)

        # 将图像数据转换为torch.Tensor（保持不变）
        input = torch.Tensor(input).unsqueeze(0).cuda() * 2 - 1
        cond = torch.Tensor(cond).unsqueeze(0).cuda() * 2 - 1 if cond is not None else None
        target = torch.Tensor(target).unsqueeze(0).cuda() * 2 - 1 if target is not None else None

        inputs.append(input)
        conds.append(cond)
        targets.append(target)
        original_dcm_mins.append(original_min)
        original_dcm_maxs.append(original_max)
        original_dcm_metas.append(dcm_meta)

        # 堆叠tensors（保持不变）
    inputs = torch.stack(inputs, dim=0)
    conds = torch.stack(conds, dim=0) if conds else None
    targets = torch.stack(targets, dim=0) if targets else None

    # 模型加载和去噪处理部分
    print("Loading model and performing denoising")
    diffusion = Model.create_model(model_cfg)
    timesteps = list(range(0, 500, 20)) + list(range(500, 2000, 500)) + [1999]

    for i in range((length - 1) // bs + 1):
        input = inputs[i * bs:i * bs + bs if i * bs + bs < length else length, ...]
        n = input.shape[0]
        cond = conds[i * bs:i * bs + bs if i * bs + bs < length else length, ...] if conds is not None else None

        input = torch.cat([input] * mean_num)
        cond = torch.cat([cond] * mean_num) if cond is not None else None

        diffusion.inversion(input, cond, timesteps, ddim_eta, batch_size=n * mean_num, ddim=if_ddim,
                            lambda1=torch.full(input.shape, lam0).to(input.device), a=a, b=b, c=c, resume=resume,
                            mode=mode, continous=False)

        visuals = diffusion.get_current_visuals(sample=True)

        # 保存去噪后的图像为DCM格式（保留原样，但确保在正确的缩进级别）
        for j in range(n):
            denoised = torch.mean(visuals['SAM'][-(n * mean_num - j)::n, ...], dim=0)
            denoised_npy = denoised.clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5  # 假设模型输出在[-1, 1]范围内

            if target_folder is not None:
                input_npy = inputs[i * bs + j, ...].clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5
                target_npy = targets[i * bs + j, ...].clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5
                psnr_org = metrics.calculate_psnr(input_npy * 255., target_npy * 255.)
                psnr = metrics.calculate_psnr(denoised_npy * 255., target_npy * 255.)
                ssim_org = metrics.calculate_ssim(input_npy * 255., target_npy * 255.)
                ssim = metrics.calculate_ssim(denoised_npy * 255., target_npy * 255.)
                # print('%s-resolution %d, psnr %.2f to %.2f; ssim %.3f to %.3f' % (
                #     low_imgs[i * bs + j], res, psnr_org, psnr, ssim_org, ssim))

            # 保存去噪后的图像为DCM格式
            output_path = os.path.join(root, output_folder,
                                       f"{os.path.splitext(low_dcm_files[i * bs + j])[0]}_denoised.dcm")
            dcm_meta =  original_dcm_metas[i * bs + j]  # 获取原始DCM的元信息
            original_min = original_dcm_mins[i * bs + j]
            original_max = original_dcm_maxs[i * bs + j]

            save_dcm_img(denoised_npy, output_path, original_min, original_max, dcm_meta)

            # 注意：确保save_dcm_img函数能够正确处理DCM文件的保存，并且所有其他与图像处理相关的代码都已根据DCM格式进行了适当的修改。