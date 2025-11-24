# 参考https://github.com/DengPingFan/PraNet/blob/master/lib/PraNet_Res2Net.py#L105  中RA的使用


from VM_UNet.models.vmunet.vmamba import VSSM
import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
        
        
        
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Enlarge the feature size: x2
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(x1) * x2   # self.upsample(x1)
        # # 将 x1 上采样后与 x2 上采样、x3 相乘的结果相乘
        x3_1 = self.conv_upsample2(self.upsample(x1)) * self.conv_upsample3(self.upsample(x2)) * x3
        # x3_1 = self.conv_upsample2(self.upsample(x1)) \
        #        * self.conv_upsample3(x2) * x3

        # 将 x2_1 和 x1_1 在第2个通道上进行通道合并
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1)   # self.upsample(x1_1)
        # # 对合并后的结果进行卷积操作
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x



class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 channel= 96,       # 32
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
         # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(384, channel)        # 512  # 192
        self.rfb3_1 = RFB_modified(768, channel)        #1024  # 384
        self.rfb4_1 = RFB_modified(768, channel)        # 2048
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(768, 192, kernel_size=1)      # 2048, 256
        self.ra4_conv2 = BasicConv2d(192, 192, kernel_size=5, padding=2)        # 256, 256
        self.ra4_conv3 = BasicConv2d(192, 192, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(192, 192, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(192, 1, kernel_size=1)    # to get skip_list[0] = Tensor[1,64,64,96]
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(768, 96, kernel_size=1)       # 1024, 64
        self.ra3_conv2 = BasicConv2d(96, 96, kernel_size=3, padding=1)  # 64, 64
        self.ra3_conv3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(96, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(384, 96, kernel_size=1)     # 512, 64
        self.ra2_conv2 = BasicConv2d(96, 96, kernel_size=3, padding=1)   # 64, 64
        self.ra2_conv3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(96, 1, kernel_size=3, padding=1)

    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # logits = self.vmunet(x)
        # if self.num_classes == 1: return torch.sigmoid(logits)
        # else: return logits
        skip_list = []
        x = self.vmunet.patch_embed(x)
        if self.vmunet.ape:
            x = x + self.vmunet.absolute_pos_embed
        x = self.vmunet.pos_drop(x)    # channel=96

        # ---- low-level features ----
        x1 = self.vmunet.layers[0](x)      # resnet.layer1: bs, 256, 88, 88    but vmunet.layers[0].output_channel=192
        x2 = self.vmunet.layers[1](x1)     # resnet.layer2: bs, 512, 44, 44    but vmunet.layers[1].output_channel=384

        x3 = self.vmunet.layers[2](x2)     # bs, 1024, 22, 22           but vmunet.layers[1].output_channel=768
        x4 = self.vmunet.layers[3](x3)     # bs, 2048, 11, 11           but vmunet.layers[1].output_channel=768
        x2_rfb = self.rfb2_1(x2.permute(0, 3, 1, 2))        # x2
        x3_rfb = self.rfb3_1(x3.permute(0, 3, 1, 2))        # x3
        x4_rfb = self.rfb4_1(x4.permute(0, 3, 1, 2))        # x4

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        # scale_factor=8 will get lateral_map_4=Tensor(1,1,128,128), but to make lateral_map_4=Tensor(1,1,192,192),scale_factor=12
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=4, mode='bilinear')
        lateral_map_5 = lateral_map_5.repeat(1,96,1,1)
        skip_list.append(lateral_map_5.permute(0, 2, 3, 1))
        
         # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.5, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1      # x= Tensor (1,1,8,8)
        x = x.permute(0, 3, 2, 1)     #    x4=Tensor(1,8,8,768)
        # torch.expand 只能对维度值包含1的张量Tensor进行扩展，无需扩展的维度保持不变或置-1。
        x = x.expand(-1, -1, -1, 768).mul(x4)   # (-1, 2048, -1, -1)
        x = x.permute(0, 3, 1,2)
        x = self.ra4_conv1(x)  # 1,192,8,8

        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        lateral_map_4 = x.repeat(1, 1, 4, 4)
        skip_list.append(lateral_map_4.permute(0, 2, 3, 1))
        x = ra4_feat + crop_4
        # # scale_factor=32 will get lateral_map_4=Tensor(1,1,256,256), but to make lateral_map_4=Tensor(1,1,384,384),scale_factor=24
        # lateral_map_4 = F.interpolate(x, scale_factor=4, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        # skip_list.append(lateral_map_4.permute(0, 2, 3, 1))

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=1, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        # x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = x.permute(0, 3, 2, 1)     #    x3=Tensor(1,8,8,768)
        x = x.expand(-1, -1, -1, 768).mul(x3)   # (-1, 2048, -1, -1)
        x = x.permute(0, 3, 1,2)
        x = self.ra3_conv1(x)   # 1,96,8,8
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        lateral_map_3 = x.repeat(1, 4, 2, 2)
        skip_list.append(lateral_map_3.permute(0, 2, 3, 1))
        # lateral_map_3 = F.interpolate(x, scale_factor=2, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)


        x = ra3_feat + crop_3
        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        # x = x.expand(-1, 512, -1, -1).mul(x2)
        x = x.permute(0, 3, 2, 1)     #    x2=Tensor(1,16,16,384)
        x = x.expand(-1, -1, -1, 384).mul(x2)   # (-1, 2048, -1, -1)
        x = x.permute(0, 3, 1,2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        lateral_map_2 = x.repeat(1, 8, 1, 1)
        x = ra2_feat + crop_2
        # lateral_map_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        skip_list.append(lateral_map_2.permute(0, 2, 3, 1))

        x = self.vmunet.forward_features_up(lateral_map_2.permute(0, 2, 3, 1), skip_list)
        # x = self.vmunet.forward_features_up(x, skip_list)     # x:Tensor(1,1,16,16);   skip_list[Tensor(1,1,128,128),(1,1,256,256),(1,1,128,128),(1,1,128,128)]
        x = self.vmunet.forward_final(x)
        logits = x
        if self.num_classes == 1: return torch.sigmoid(logits)
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