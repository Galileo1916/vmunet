from .mamba_vision import MambaVision, mamba_vision_T, MambaVision_sim
import torch
from torch import nn
import torch.nn.functional as F

# by mingya zhang  dg20330034@smail.nju.edu.cn 2024 08 16

class SobelEdgeDetection(nn.Module):
    def __init__(self):
        super(SobelEdgeDetection, self).__init__()
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

class HMTUNet(nn.Module):
    
    def __init__(self, 
                    input_channels=3,
                    num_classes=1,
                    depths=[1, 3, 8, 4],
                    num_heads=[2, 4, 8, 16],
                    window_size=[8, 8, 14, 7],
                    dim=80,
                    in_dim=32,
                    mlp_ratio=4,
                    resolution=224,
                    drop_path_rate=0.2,
                    load_ckpt_path=None,
                    **kwargs):
        
        super().__init__()
        
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        
        self.hmtunet = MambaVision_sim(
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                dim=dim,
                in_dim=in_dim,
                mlp_ratio=mlp_ratio,
                resolution=resolution,
                drop_path_rate=drop_path_rate,
        )
        self.edge_detection_module = SobelEdgeDetection()
        
        
        
    
    def forward(self, x):
        # todo: For Sobel edge detection
        # vmunet 先提取语义特征
        semantic_logits = self.hmtunet(x)  # Tensor(1,1,256,256), 包含了一系列上采样和下采样的操作
        # 提取边缘特征
        edge_logits = self.edge_detection_module(x)
        # 归一化到[-1, 1]范围
        max_val, min_val = torch.max(edge_logits), torch.min(edge_logits)
        normalized_map = (edge_logits - min_val) / (max_val - min_val)
        normalized_edge_map = 2 * normalized_map - 1

        # 将边缘特征与高层语义特征融合
        # fused_logits = torch.cat([semantic_logits, normalized_edge_map], dim=1)  # 按通道维度连接
        fused_logits = torch.stack([semantic_logits, normalized_edge_map], dim=1).mean(dim=1)
        return fused_logits

    
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.hmtunet.state_dict()
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = model_checkpoint['state_dict']
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict) 
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.hmtunet.load_state_dict(model_dict)
            
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.hmtunet.state_dict()
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_order_dict = model_checkpoint['state_dict']
            pretrained_dict = {}
            for k,v in pretrained_order_dict.items():
                if 'levels.0' in k:
                    new_k = k.replace('levels.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'levels.1' in k: 
                    new_k = k.replace('levels.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'levels.2' in k: 
                    new_k = k.replace('levels.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'levels.3' in k: 
                    new_k = k.replace('levels.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
                    
            # decoder 
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.hmtunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
            
    
    

        
    