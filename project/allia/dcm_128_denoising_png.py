import torch
import model as Model
import argparse
import os
import numpy as np
import yaml
import json
from collections import OrderedDict
import cv2
import pydicom  # 导入处理DCM文件的库
import metrics
import imageio.v2 as imageio


# 读取DCM图片的函数（保留原样）
def read_dcm_img(path):
    """读取DCM图片并返回归一化后的numpy数组"""
    print(f"Reading DCM image from {path}")
    dcm = pydicom.dcmread(path)
    print(f"DCM image loaded with shape {dcm.pixel_array.shape}")
    pixel_array = dcm.pixel_array.astype(np.float32)
    normalized_pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    print(f"DCM image normalized to range [0, 1]")
    return normalized_pixel_array

#
# # 读取PNG图片的函数（稍作修改，直接返回[0, 1]范围的numpy数组）
# def read_a_img(path):
#     """读取PNG图片并返回[0, 1]范围的numpy数组"""
#     print(f"Reading PNG image from {path}")
#     return imageio.imread(path) / 255.0
#
# def save_a_img(img, path): # function of save $img$ (np.ndarray of range [0.0, 1.0]) to $path$
#     return imageio.imwrite(path, (img*65535).astype(np.uint16))

# # 保存为DCM格式的函数（新添加，但在这个示例中不使用）
# def save_dcm_img(img, path, original_size=(512, 512)):
#     """保存numpy数组为DCM文件，并假设原始DCM是original_size大小，像素范围在65535以内"""
#     # 此函数在本示例中不使用，因此详细实现略去
#     pass
def save_dcm_img(img, path):
    print("Saving DCM image to path:", path)
    # 假设img是[0, 1]范围的numpy数组，需要转换为uint16类型，范围0-65535
    pixel_array = (img * 65535).astype(np.uint16)

    # 创建一个新的DCM数据集对象
    dcm = pydicom.Dataset()
    dcm.Rows, dcm.Columns = 512, 512  # 设置尺寸为512x512
    dcm.BitsAllocated = 16
    dcm.BitsStored = 16
    dcm.HighBit = 15
    dcm.PixelRepresentation = 0
    dcm.PhotometricInterpretation = "MONOCHROME2"
    dcm.PixelData = pixel_array.tobytes()  # 设置像素数据
    dcm.is_little_endian = True  # 假设使用小端字节序
    dcm.is_implicit_VR = True  # 假设使用隐式VR
    # 使用pydicom.dcmwrite来保存DCM文件，它会自动处理文件元数据和传输语法
    pydicom.dcmwrite(path, dcm)


# 假设的确定文件类型和尺寸的函数（新添加）
def determine_file_type_and_size(path):
    """检测文件类型（DCM或PNG）和尺寸"""
    _, ext = os.path.splitext(path)
    if ext.lower() == '.dcm':
        dcm = pydicom.dcmread(path)
        return 'dcm', dcm.Rows, dcm.Columns
    elif ext.lower() == '.png':
        img = imageio.imread(path)
        return 'png', img.shape[0], img.shape[1]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # 主程序部分


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
    res = cfg['data']['res']  # 输出图像的尺寸
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
    inputs, conds, targets = [], [], []
    original_dcm_sizes = []  # 用于存储DCM文件的原始尺寸信息（在这个示例中不使用）
    if length == -1:
        length = len(low_dcm_files)

    for i, low_name in enumerate(low_dcm_files):
        file_path = os.path.join(root, input_folder, low_name)
        file_type, height, width = determine_file_type_and_size(file_path)
        print(f"Processing file {low_name} of type {file_type} with size {height}x{width}")

        if file_type != 'dcm':
            print(f"Skipping non-DCM file: {low_name}")
            continue

        if height != 512 or width != 512:
            print(f"Skipping non-standard size DCM: {low_name}")
            continue

        input = read_dcm_img(file_path)
        # print(input)
        input = cv2.resize(input, (res, res), cv2.INTER_CUBIC)
        print(f"Resized input image to {res}x{res}")

        # ...（处理conds和targets的逻辑保持不变，但注意也要转换为torch.Tensor）...
        # cond = torch.Tensor(cond).unsqueeze(0).cuda() * 2 - 1 if cond_folder is not None else None
        # target = torch.Tensor(target).unsqueeze(0).cuda() * 2 - 1 if target_folder is not None else None

        # 将图像数据转换为torch.Tensor并添加到inputs列表
        input = torch.Tensor(input).unsqueeze(0).cuda() * 2 - 1
        inputs.append(input)

        # ...（处理conds和targets的类似转换和添加）...
        # conds.append(cond)
        # targets.append(target)
    # 堆叠inputs, conds, targets等
    print("Stacking input tensors")
    inputs = torch.stack(inputs, dim=0)
    conds = torch.stack(conds, dim=0) if cond_folder is not None else None
    if target_folder is not None and targets:  # 确保targets列表非空
        targets = torch.stack(targets, dim=0) if target_folder is not None else None
    else:
        targets = None


    # 模型加载和去噪处理部分
    print("Loading model and performing denoising")
    diffusion = Model.create_model(model_cfg)
    timesteps = list(range(0, 500, 20)) + list(range(500, 2000, 500)) + [1999]
    # print("length=",length)
    for i in range((length - 1) // bs + 1):

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

        # 保存结果部分
        for j in range(n):
            denoised = torch.mean(visuals['SAM'][-(n * mean_num - j)::n, ...], dim=0)
            denoised_npy = denoised.clamp_(-1, 1).squeeze(0).cpu().numpy() * 0.5 + 0.5

            output_path = os.path.join(root, output_folder,
                                       f"{os.path.splitext(low_dcm_files[i * bs + j])[0]}_denoised.png")
            print(f"Saving denoised image to {output_path}")
            save_a_img(denoised_npy, output_path)
