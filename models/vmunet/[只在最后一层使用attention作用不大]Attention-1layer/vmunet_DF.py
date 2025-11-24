from VM_UNet.models.vmunet.vmamba import VSSM
import torch
from torch import nn
import torch.nn.functional as F

# https://github.com/c-feng/DirectionalFeature/tree/master
# 通过方向场 （DF） 模块，学习从最近的组织边界指向每个像素的方向场，以增加类间差异和类内不一致

class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n=5, n_class=1, auxseg=False):
        super(SelFuseFeature, self).__init__()

        self.shift_n = shift_n
        self.n_class = n_class
        self.auxseg = auxseg
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True),
                                       )
        if auxseg:
            self.auxseg_conv = nn.Conv2d(in_channels, self.n_class, 1)

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        scale = 1.

        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        # 添加了一个位移（displacement）到网格上。这个位移由变量 scale 控制，加上了一个名为 df 的位移场。结果是一个新的网格，表示应用位移后的坐标。
        grid = grid + scale * df

        # 将维度进行了置换，得到了形状为 (N, H, W, 2) 的网格
        grid = grid.permute(0, 2, 3, 1)     #.transpose(1, 2)
        grid_ = grid + 0.
        grid[..., 0] = 2 * grid_[..., 0] / H - 1      # 将 x 和 y 坐标归一化到范围 [-1, 1] 内,归一化后的坐标用于采样
        grid[..., 1] = 2 * grid_[..., 1] / W - 1

        # features = []
        select_x = x.clone()
        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')    # 使用 grid 提供的坐标从 select_x 中采样值。结果是根据指定的采样方法和填充模式对 select_x 进行了变换。
            # features.append(select_x)
        # select_x = torch.mean(torch.stack(features, dim=0), dim=0)
        # features.append(select_x.detach().cpu().numpy())
        # np.save("/root/chengfeng/Cardiac/source_code/logs/acdc_logs/logs_temp/feature.npy", np.array(features))
        if self.auxseg:
            auxseg = self.auxseg_conv(x)
        else:
            auxseg = None

        select_x = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return [select_x, auxseg]


class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 shift_n=5,
                 auxseg=False,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )

        # Direct Field
        self.ConvDf_1x1 = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)
        self.SelDF = SelFuseFeature(256, auxseg=auxseg, shift_n=shift_n)
        self.Conv_1x1 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)   # num_class=1

    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x = self.vmunet(x)   # Tensor(1,1,256,256)

        # Direct Field
        x = x.permute(2,3,1,0)
        df = self.ConvDf_1x1(x)
        # df = None

        x_auxseg = self.SelDF(x, df)
        x, auxseg = x_auxseg[:2]
        logits = x.permute(3,2,0,1)

        # df = F.interpolate(df, size=x.shape[-2:], mode='bilinear', align_corners=True)
        #logits = self.Conv_1x1(x)

        if self.num_classes == 1:
            L = torch.sigmoid(logits)
            print(L)
            return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['state_dict']      # modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['state_dict']     # modelCheckpoint['model']
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
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")