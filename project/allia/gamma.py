import numpy as np
import os
from PIL import Image
import pydicom
import matplotlib.pyplot as plt

#全剂量照片
folder_path_full = r"E:\yidatest3\test_data\Liver\full"
file_name_full = r"1-200.dcm"
file_path_full = os.path.join(folder_path_full, file_name_full)
ds_full = pydicom.dcmread(file_path_full)
img_full = ds_full.pixel_array
# img_full = np.array(Image.open(file_path_full))

#低剂量照片
folder_path_low = r"E:\yidatest3\test_data\Liver\low"
# folder_path_low = r"E:\yidatest3\test_data\Liver\hr_dn"
file_name_low = r"1-200.dcm"
file_path_low = os.path.join(folder_path_low, file_name_low)
ds_low = pydicom.dcmread(file_path_low)
img_low = ds_low.pixel_array
# img_low = np.array(Image.open(file_path_low))

def gamma(img_f, img_l, dd, dta, ps):
    """
    绝对剂量伽马分析

    参数:
    img_f: 参考剂量分布图（二维NumPy数组）
    img_l: 计算剂量分布（二维NumPy数组）
    dd: 剂量差异标准（百分比）
    dta: DTA标准（单位通常为mm）
    ps: 像素间距（单位通常为mm）

    返回:
    dis: gamma因子分布（二维NumPy数组）
    r: 通过率（浮点数）
    """

    # 转换为浮点数以避免后续计算中的精度问题
    max_img_f = np.max(img_f)  # 获取img中的最大值
    print("max_img_f=",max_img_f)
    img_f = img_f.astype(np.float64)  # 将img转换为64位浮点数数组
    img_l = img_l.astype(np.float64)  # 将I转换为64位浮点数数组
    dd = np.float64(dd)  # 将ra转换为64位浮点数
    dta = np.float64(dta)  # 将dta转换为64位浮点数
    ps = np.float64(ps)  # 将ps转换为64位浮点数

    # 获取图像的大小
    sx, sy = img_f.shape  # 获取img的行数和列数

    # 初始化gamma因子分布数组，大小与img相同
    dis = np.zeros((sx, sy))  # 创建一个与img相同大小的二维零数组

    # 将剂量差异标准和DTA标准转换为实际数值
    detaD = dd * 0.01 * max_img_f  # 剂量差异标准转换为绝对剂量值
    # detaD = dd * 0.01
    # detad = dta * ps  # DTA标准转换为像素单位
    detad = dta

    # 统计有剂量的区域，排除体表外区域（这里假设是10%的阈值）

    area = img_f > (max_img_f * 0.10)  # 创建一个与img相同大小的布尔数组，标记有剂量的区域
    # plt.imshow(area, cmap='gray')  # 使用灰度图来显示布尔数组
    # plt.show()
    # print(area)
    # # 统计area数组中True的数量
    # num_dose_pixels = np.sum(area)
    # print(f"Number of dose pixels: {num_dose_pixels}")

    # 遍历每个像素
    for i in range(1, sx ):
        for j in range(1, sy ):
            if area[i - 1, j - 1] == 1:
                # 对10个像素内的计算, ms, md 行的起止处，ns, nd 列的起止处
                ms = max(1, i - 2)
                ns = max(1, j - 2)
                md = min(sx, i + 2)
                nd = min(sy, j + 2)
    # for i in range(sx):  # 遍历图像的行
    #     for j in range(sy):  # 遍历图像的列
    #         if area[i, j]:  # 如果当前像素在有剂量的区域内
    #             # if area[i, j] != True:
    #             #     print("i=",i," j=",j,'area=',area[i, j])
    #
    #
    #             # 计算当前像素附近范围内的gamma值
    #             ms = max(1, i - 1)  # 计算行起始位置（防止超出边界）
    #             ns = max(1, j - 1)  # 计算列起始位置（防止超出边界）
    #             md = min(sx, i + 2)  # 计算行结束位置（防止超出边界）
    #             nd = min(sy, j + 2)  # 计算列结束位置（防止超出边界）

                # 初始化数组用于存储距离、剂量差异和gamma值
                d = np.zeros((md - ms, nd - ns))  # 存储距离
                D = np.zeros((md - ms, nd - ns))  # 存储剂量差异
                ga = np.zeros((md - ms, nd - ns))  # 存储gamma值

                # 计算距离和剂量差异
                for k in range(ms, md):  # 遍历附近像素的行
                    for p in range(ns, nd):  # 遍历附近像素的列
                        kk, pp = k - ms, p - ns  # 转换为以当前像素为原点的坐标


                        d[kk, pp] = np.sqrt((k - i) ** 2 + (p - j) ** 2) * ps  # 计算距离
                        # print("d = ", d)
                        # D[kk, pp] = (img_l[k, p] - img_f[i, j]) / max_img_f  # 计算剂量差异
                        D[kk, pp] = img_l[k, p] - img_f[i, j]  # 计算剂量差异
                        # print("D = ", D)

                        # 计算gamma值
                        # ga[kk, pp] = np.sqrt(D[kk, pp] ** 2 / detaD ** 2)
                        ga[kk, pp] = np.sqrt(d[kk, pp] ** 2 / detad ** 2 + D[kk, pp] ** 2 / detaD ** 2)
                        # print("ga = ", ga[kk, pp])
                        # print("sx=", sx, " sy=", sy, " i=", i, " j=", j, " k=", k, " p=", p, " ms=", ms, " ns=", ns, " md=", md," nd=", nd," kk=",kk," pp=",pp," d=",d[kk,pp] ," D=",D[kk,pp]," ga=",ga)
                # 找到最小gamma值并存储在Ir中
                dis[i, j] = np.min(ga)
                # print("Ir = ",minnum)
    # 计算通过率：gamma值大于1的像素数与有剂量像素数的比值（1减去该比值）
    r = 1 - np.sum(dis > 1) / np.sum(area)
    # return r,dis

    return r
result = gamma(img_full,img_low,3,3,0.7421875)
print("Gamma通过率:",result)

