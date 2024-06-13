import torch
import model as Model
import argparse
import os
import numpy as np
import yaml
import json
from collections import OrderedDict
import pydicom
import cv2


# 读取DCM图片的函数
def read_dcm_img(path):
    print(f"Reading DCM image from {path}")
    dcm = pydicom.dcmread(path)

    print(f"DCM image loaded with shape {dcm.pixel_array.shape}")
    pixel_array = dcm.pixel_array.astype(np.float32)
    min_val, max_val = pixel_array.min(), pixel_array.max()
    normalized_pixel_array = (pixel_array - min_val) / (max_val - min_val)
    print(f"DCM image normalized to range [0, 1]")
    dcm_meta = dcm  # 返回原始的dcm对象，包含所有元信息
    return normalized_pixel_array, min_val, max_val, dcm_meta


# 保存为DCM格式的函数
def save_dcm_img(img, path, original_min, original_max, dcm_meta):
    print(f"Saving DCM image to {path}")
    pixel_array = (img * (original_max - original_min) + original_min).astype(np.uint16)
    pixel_array = pixel_array.clip(0, 65535).astype(np.uint16)

    # 复制除了PixelData之外的所有元信息
    # output_dcm = pydicom.Dataset()
    # for elem in dcm_meta:
    #     if elem.keyword != 'PixelData':
    #         output_dcm[elem.keyword] = elem.value  # 使用字典接口添加元素

    # 使用 copy() 方法复制除了 PixelData 之外的所有元信息
    output_dcm = dcm_meta.copy()
    output_dcm.PixelData = pixel_array.tobytes()  # 更新 PixelData
    # del output_dcm.PixelData  # 删除原有的 PixelData，因为我们要添加新的

    # 如果需要，设置其他必要的元信息（例如，如果原始图像被调整了大小）
    output_dcm.Rows, output_dcm.Columns = pixel_array.shape

    # 保存DICOM文件
    pydicom.filewriter.dcmwrite(path, output_dcm)

    # dcm = pydicom.Dataset()
    # dcm.PixelData = pixel_array.tobytes()
    # dcm.Rows, dcm.Columns = img.shape
    # dcm.BitsAllocated = 16
    # dcm.BitsStored = 16
    # dcm.HighBit = 15
    # dcm.PixelRepresentation = 0
    # dcm.PhotometricInterpretation = "MONOCHROME2"
    # dcm.is_little_endian = True  # 假设使用小端字节序
    # dcm.is_implicit_VR = True  # 假设使用隐式VR
    # # 设置字节序和VR（值表示）类型，这里假设使用小端字节序和显式VR
    # pydicom.dcmwrite(path, dcm)



if __name__ == "__main__":
    # 参数设置部分
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Dn_Liver_128.yaml',
                        help='yaml file for configuration')
    args = parser.parse_args()
    print(f"Loading configuration from {args.config}")
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

    # 数据处理部分
    print(f"Processing input files from {os.path.join(root, input_folder)}")
    low_dcm_files = sorted(os.listdir(os.path.join(root, input_folder)))
    cond_dcm_files = sorted(os.listdir(os.path.join(root, cond_folder))) if cond_folder is not None else None
    target_imgs = sorted(os.listdir(os.path.join(root, target_folder))) if target_folder is not None else None
    print(f"Found {len(low_dcm_files)} input files")
    if cond_folder is not None:
        print(f"Found {len(cond_dcm_files)} condition files")
    if length == -1:
        length = len(low_dcm_files)

    # 创建输出文件夹（如果不存在）
    print(f"Creating output folder {os.path.join(root, output_folder)} if it doesn't exist")
    os.makedirs(os.path.join(root, output_folder), exist_ok=True)

    inputs, conds, targets = [], [], []
    original_dcm_mins, original_dcm_maxs, original_dcm_metas = [], [], []
    if cond_folder is not None:
        original_cond_mins, original_cond_maxs = [], []

    # 读取并处理输入图像和条件图像
    for i, low_name in enumerate(low_dcm_files):
        # 读取并处理输入图像
        file_path = os.path.join(root, input_folder, low_name)
        input_img, original_min, original_max, dcm_meta = read_dcm_img(file_path)
        input_img = cv2.resize(input_img, (res, res), cv2.INTER_CUBIC)

        # 读取并处理条件图像（如果有的话）
        if cond_folder is not None:
            cond_path = os.path.join(root, cond_folder, cond_dcm_files[i])
            cond_img, original_cond_min, original_cond_max, dcm_meta = read_dcm_img(cond_path)
            cond_img = cv2.resize(cond_img, (res, res), cv2.INTER_CUBIC)
        else:
            cond_img = None
            original_cond_min, original_cond_max = None, None

        if target_folder is not None:
            target_path = os.path.join(root, target_folder, target_imgs[i])
            try:
                target, original_min, original_max, dcm_meta = read_dcm_img(target_path)
                target = cv2.resize(target, (res, res), cv2.INTER_CUBIC)
            except Exception as e:
                print(f"Error processing target image at {target_path}: {e}")
                target = None  # 或者设置为其他默认值，或者抛出异常
        else:
            target = None

        # 将图像数据转换为torch.Tensor并添加到列表中
        input_tensor = torch.Tensor(input_img).unsqueeze(0).cuda() * 2 - 1
        cond_tensor = torch.Tensor(cond_img).unsqueeze(0).cuda() * 2 - 1 if cond_img is not None else None
        inputs.append(input_tensor)
        conds.append(cond_tensor)
        original_dcm_mins.append(original_min)
        original_dcm_maxs.append(original_max)
        original_dcm_metas.append(dcm_meta)
        if cond_img is not None:
            original_cond_mins.append(original_cond_min)
            original_cond_maxs.append(original_cond_max)

    # 堆叠inputs, conds, targets等
    print("Stacking input tensors")
    inputs = torch.stack(inputs, dim=0)
    conds = torch.stack(conds, dim=0) if cond_folder is not None else None
    if target_folder is not None and targets:  # 确保 targets 非空
        targets = torch.stack(targets, dim=0)
    else:
        targets = None  # 或者根据你的需要设置其他默认值

    # 模型加载和去噪处理部分
    print("Loading model and performing denoising")
    diffusion = Model.create_model(model_cfg)
    timesteps = list(range(0, 500, 20)) + list(range(500, 2000, 500)) + [1999]

    # ...（去噪处理的逻辑保持不变，但注意保存结果部分需要修改）...

    # 保存结果部分
    for i in range((length-1)//bs+1):
        input = inputs[i * bs:i * bs + bs if i * bs + bs < length else length, ...]
        n = input.shape[0]
        cond = conds[i * bs:i * bs + bs if i * bs + bs < length else length, ...] if cond_folder is not None else None
        input = torch.cat([input] * mean_num)
        cond = torch.cat([cond] * mean_num) if cond_folder is not None else None

        diffusion.inversion(input, cond, timesteps, ddim_eta,
                            batch_size=n * mean_num, ddim=if_ddim,
                            lambda1=torch.full(input.shape, lam0).to(input.device), a=a, b=b, c=c, resume=resume,
                            mode=mode, continous=False)

        visuals = diffusion.get_current_visuals(sample=True)

        for j in range(n):
            denoised = torch.mean(visuals['SAM'][-(n * mean_num - j)::n, ...], dim=0)
            denoised_npy = denoised.clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5

            # 使用原始像素值范围转换回uint16并保存为DICOM格式
            original_min = original_dcm_mins[i * bs + j]
            original_max = original_dcm_maxs[i * bs + j]
            dcm_meta = original_dcm_metas[i * bs + j]  # 假设original_dcm_metas保存了所有输入的元信息

            denoised_dcm = (denoised_npy * (original_max - original_min) + original_min).clip(0, 65535).astype(
                np.uint16)

            output_path = os.path.join(root, output_folder,
                                       os.path.splitext(low_dcm_files[i * bs + j])[0] + '_denoised.dcm')
            save_dcm_img(denoised_dcm, output_path, original_min, original_max, dcm_meta)
