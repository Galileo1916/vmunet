# 【弃】改成使用vmunetTest_after_trained.py
# -*- coding: utf-8 -*-
# 输入测试集图片路径和最优训练的pth，直接输出测试集的预测结果图片和mIOU
# Test Cartilage_endoscope _ cannot used
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
from tqdm import tqdm
from utils_metrics import compute_mIoU, show_results
import numpy as np
import torch
import os
import cv2
# from models.vmunet.vmunet import VMUNet
from models.vmunet.vmunet_Sobel import VMUNet_Sobel as VMUNet
from configs.config_setting import setting_config


def cal_miou(test_dir="/home/data/glw/Projects/CartilageData_endoscope/img_dir/val",
             pred_dir=" ", gt_dir="", best_pth=''):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 1
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["_background_", "wear"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对

    # 加载模型

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        config = setting_config
        model_cfg = config.model_config
        net = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        # net.load_from()
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load(best_pth, map_location=device),False) # todo    # 加载的model与原pretrained model权重不一致,strict=False
        # 测试模式
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        # for image_id in tqdm(image_ids):
        #     # image_path = os.path.join(test_dir, image_id + ".jpg")   #'.png'
        #     for extension in ['.jpg', '.png']:
        #         image_path = os.path.join(test_dir, image_id + extension)
        #         if os.path.exists(image_path):
        #             break
        #     img = cv2.imread(image_path)
        #     origin_shape = img.shape
        #     # print(origin_shape)
        #     # 转为灰度图
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #     img = cv2.resize(img, (256, 256))    #todo: VMUNet  size
        #     # 转为batch为1，通道为1，大小为512*512的数组
        #     img = img.reshape(1, 1, img.shape[0], img.shape[1])
        #     # 转为tensor
        #     img_tensor = torch.from_numpy(img)
        #     # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        #     img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        #     # 预测
        #     pred = net(img_tensor)
        #     # 提取结果
        #     pred = np.array(pred.data.cpu()[0])[0]
        #     pred[pred >= 0.1] = 255
        #     pred[pred < 0.1] = 0
        #     pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes+1, # todo: +1?
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    test_dir = "/data/glw/Projects/mmsegmentation/projects/KneeCartilage02_UNet/CartilageData_202409_202411/img_dir/val/"
    gt_dir = "/data/glw/Projects/mmsegmentation/projects/KneeCartilage02_UNet/CartilageData_202409_202411/mask_dir/val"
    best_pth = "/data/glw/Projects/VM_UNet/results/vmunet_CSAM_Sobel_cartilage_Friday_20_December_2024_06h_09m_51s/checkpoints/best.pth"
    pred_dir = '/data/glw/Projects/VM_UNet/results/vmunet_CSAM_Sobel_cartilage_Friday_20_December_2024_06h_09m_51s/outputs/'   # save pred results
    cal_miou(test_dir,pred_dir,gt_dir,best_pth)