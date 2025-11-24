from VM_UNet.LOD_project.vmamba_LOD import VSSM
import torch
from torch import nn
import torch.nn.functional as F
from VM_UNet.LOD_project.utils_LOD import BCELoss
from VM_UNet.LOD_project.config_setting_LOD import setting_config
from torch.utils.data import DataLoader
from VM_UNet.datasets.dataset import NPY_datasets
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec


class LODEdgeDetection(nn.Module):
    def __init__(self):
        super(LODEdgeDetection, self).__init__()
        # 预定义的八个方向的 Sobel 滤波器或者其他导数计算方式
        self.sobel_filters = [
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),
            # 其他方向的滤波器...
        ]

    def forward(self, x):
        # 对输入图像 x 进行边缘检测
        edge_maps = []
        for sobel_filter in self.sobel_filters:
            sobel_filter = sobel_filter.unsqueeze(0).unsqueeze(0).to(x.device)  # 调整维度以适应卷积操作
            # 重复Sobel滤波器以匹配输入特征的通道数
            sobel_filter = sobel_filter.repeat(1, 3, 1, 1)
            edge_map = F.conv2d(x, sobel_filter, padding=1)
            edge_maps.append(edge_map)

        # 选择每个像素点导数最大方向
        edge_map = torch.max(torch.stack(edge_maps, dim=0), dim=0)[0]

        return edge_map


class VMUNet_LOD(nn.Module):   ### 【无法运行】 只适用于上采样和下采样过程中通道数不变的场景下 !!!
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 num_directions=8,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.iter = 0  # 初始化计数器
        self.num_directions = num_directions

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )
        self.edge_detection_module = LODEdgeDetection()
        self.gt_masks = self.get_gt_masks()

    def forward(self, x):
        points_num = 8
        gt_masks = self.gt_masks
        self.iter = (self.iter + 1) % len(gt_masks)
        gt = gt_masks[self.iter]
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # vmunet 先提取下采样和上采样语义特征
        down_logits, up_logits = self.vmunet(x)  # Tensor(1,1,256,256), 包含了一系列上采样和下采样的操作    # pred_od, offsets

        # 提取边缘特征
        if self.training:
            # gt_masks = self.cal_gt_masks(up_logits,instances)
            od_loss = self.oriented_derivative_learning(down_logits, up_logits, gt)
            losses = {"OD_loss": od_loss}

            od_activated_map, _ = self.adaptive_thresholding(points_num, down_logits)
            border_mask_logits = self.boundary_aware_mask_scoring(gt, od_activated_map, down_logits)

            losses.update({"loss_mask": BCELoss(border_mask_logits, gt)})   # mask_rcnn_loss  -> BCEloss
            return losses
        else:
            od_activated_map, _ = self.adaptive_thresholding(points_num, down_logits)
            border_mask_logits = self.boundary_aware_mask_scoring(gt, od_activated_map, down_logits)   # down_logits
            return od_activated_map, border_mask_logits






        # 归一化到[-1, 1]范围
        max_val,  min_val = torch.max(edge_logits), torch.min(edge_logits)
        normalized_map = (edge_logits - min_val) / (max_val - min_val)
        normalized_edge_map = 2 * normalized_map - 1

        # 将边缘特征与高层语义特征融合
        # fused_logits = torch.cat([semantic_logits, normalized_edge_map], dim=1)  # 按通道维度连接
        # 将两个张量堆叠在一起，然后计算平均值
        fused_logits = torch.stack([semantic_logits, normalized_edge_map], dim=1).mean(dim=1)

        # 使用融合后的特征进行分割预测
        if self.num_classes == 1:
            return torch.sigmoid(fused_logits)
        else:
            return fused_logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']  # modelCheckpoint['model'] /  modelCheckpoint['state_dict']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']  # modelCheckpoint['model'] /  modelCheckpoint['state_dict']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")


    def get_gt_masks(self):
        config = setting_config
        train_dataset = NPY_datasets(config.data_path, config, train=True)  # MedAugment = MedAugment
        train_loader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=config.num_workers)
        val_dataset = NPY_datasets(config.data_path, config, train=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
        gt_masks = []
        for iter, data in enumerate(train_loader):
            images, targets = data
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            gt_masks.append(targets)
        return gt_masks


    def boundary_aware_mask_scoring(self, mask_logits, od_activated_map, pred_od):
        # outputs_conv = Conv2d(self.num_directions, out_channels=pred_od.size(1),kernel_size=1)
        outputs_conv = Conv2d(pred_od.size(1), out_channels=1,kernel_size=1)
        od_features = outputs_conv(pred_od)
        od_activated_map = od_activated_map.unsqueeze(dim=1)

        mask_fusion_scores = mask_logits + od_features
        border_mask_scores = ~od_activated_map * mask_logits \
                             + od_activated_map * mask_fusion_scores

        border_mask_scores = self.predictor(border_mask_scores)
        return border_mask_scores

    def oriented_derivative_learning(self, features, pred_offsets, gt):

        # gt = gt.unsqueeze(1)
        N, C, H, W = gt.shape

        pred_offsets = pred_offsets.reshape((N, H, W, -1))
        direction_nums = features.size(1)
        features_after_sample = features.new_zeros((N, direction_nums, H, W))

        grids = self.mask_reference_point(H, W, device=features.device)
        grids = grids[None, :, :, :, None]
        grids = grids.repeat(N, 1, 1, 1, direction_nums)

        grids_offsets = torch.tensor([[-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]],
                                     dtype=pred_offsets.dtype,
                                     device=pred_offsets.device)
        grids_offsets = grids_offsets.repeat(N, H, W, 1)
        offsets = pred_offsets + grids_offsets
        grids = grids + offsets

        inputs = gt
        for num in range(direction_nums):
            per_direction_grids = grids[:, :, :, :, num]
            features_after_sample[:, num:num + 1, :, :] = F.grid_sample(inputs, per_direction_grids, mode='bilinear',
                                                                        align_corners=False, padding_mode='border')

        extend_gt_masks = gt.repeat(1, direction_nums, 1, 1)

        extend_gt_masks = extend_gt_masks - features_after_sample

        oriented_gt = torch.zeros_like(extend_gt_masks)

        for num in range(direction_nums):
            offset = offsets[:, :, :, :, num]
            dis = torch.rsqrt(torch.square(offset[:, :, :, 0]) + torch.square(offset[:, :, :, 1]) + torch.ones_like(
                offset[:, :, :, 0]))
            dis = dis[:, None, :, :]
            oriented_gt[:, num:num + 1, :, :] = (extend_gt_masks[:, num:num + 1, :, :] + 1).mul(dis)

        od_loss = F.smooth_l1_loss(features, oriented_gt, reduction="mean")

        return od_loss

    def adaptive_thresholding(self, points_num, od_features):

        N, C, H, W = od_features.shape

        # for each channel, choose top points_num points and add through channel
        oriented_activated_features = torch.zeros([N, H * W], dtype=torch.float32, device=od_features.device)
        for k in range(8):
            val, idx = torch.topk(od_features.view(N, C, H * W)[:, k, :], points_num)
            for i in range(N):
                oriented_activated_features[i, idx[i]] += od_features.view(N, C, H * W)[i, k, idx[i]]

        _, idxx = torch.topk(oriented_activated_features, points_num)
        shift = H * W * torch.arange(N, dtype=torch.long, device=idxx.device)
        idxx += shift[:, None]
        activated_map = torch.zeros([N, H * W], dtype=torch.bool, device=od_features.device)
        activated_map.view(-1)[idxx.view(-1)] = True

        return activated_map.view(N, H, W), oriented_activated_features.view(N, H, W)
