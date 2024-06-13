# """Create dataset and dataloader for DICOM images"""
import logging
import torch.utils.data


# 定义一个函数来创建数据加载器
def create_dataloader(dataset, dataset_opt, phase):
    '''为给定的阶段（train或val）创建数据加载器。'''
    if phase == 'train':
        # 训练数据加载器，带有混洗和多个工作进程
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        # 验证数据加载器，不带混洗且仅使用一个工作进程
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        # 对于不支持的阶段，引发错误
        raise NotImplementedError(
            'Dataloader [{:s}] 不被支持。'.format(phase))

    # 定义一个函数来创建数据集


def create_dataset(dataset_opt, dataset_name, phase):
    '''根据提供的选项为DICOM图像创建数据集。'''
    mode = dataset_opt['mode']
    print("mode=",mode," dataset_opt=",dataset_opt," dataset_name=",dataset_name," phase=",phase)
    # 根据数据集名称导入相应的数据集类
    if dataset_name == 'uint8dataset':
        # 假设LRHRDataset已被修改为处理DICOM文件
        from data.LRHR_dataset import LRHRDataset as D
    elif dataset_name == 'uint16dataset':
        # 假设LRHRDatasetCT已被修改为处理DICOM文件
        from data.LRHR_dataset_new import LRHRDatasetCT as D
    elif dataset_name == 'uint16dataset_dn':
        # 假设NoisyCleanDatasetCT已被修改为处理DICOM文件
        from data.LRHR_dataset_new import NoisyCleanDatasetCT as D
    else:
        # 对于不支持的数据集类型，记录错误
        logging.error('无效的数据集类型')
        raise ValueError('Invalid dataset type')

    # 创建数据集对象，可能包含DICOM特定的选项（如果有的话）
    if dataset_name == 'uint16dataset_patch':
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    patch_size=dataset_opt['patch_size'],
                    patch_sample=dataset_opt['patch_sample'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LR=(mode == 'LRHR'),
                    )
    else:
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    exclude_patients=dataset_opt['exclude_patients'],
                    include_patients=dataset_opt['include_patients'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    patch_n=dataset_opt['patch_n'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LR=(mode == 'LRHR')
                    )
        # 记录数据集的创建
    logger = logging.getLogger('base')
    logger.info('数据集 [{:s} - {:s}] 已创建。'.format(dataset.__class__.__name__, dataset_opt['name']))
    # print("dataset= ", dataset)
    return dataset

# 注意：这里假设了data/LRHR_dataset.py和data/LRHR_dataset_new.py中的相应数据集类已经被修改
# 以支持DICOM图像，并接受一个'dicom'参数来启用DICOM处理逻辑。