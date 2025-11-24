import torch
import torch.nn as nn
import torchvision.models as models


# class PerceptualLoss():
#     def contentFunc(self):
#         conv_3_3_layer = 14
#         cnn = models.vgg19(pretrained=True).features
#         cnn = cnn.cuda()
#         model = nn.Sequential()
#         model = model.cuda()
#         for i, layer in enumerate(list(cnn)):
#             model.add_module(str(i), layer)
#             if i == conv_3_3_layer:
#                 break
#         return model
#
#     def __init__(self, loss):
#         self.criterion = loss
#         self.contentFunc = self.contentFunc()
#
#     def get_loss(self, fakeIm, realIm):
#         fakeIm = fakeIm.repeat(1,3,1,1)
#         realIm = realIm.repeat(1, 3, 1, 1)
#         f_fake = self.contentFunc.forward(fakeIm)
#         f_real = self.contentFunc.forward(realIm)
#         # f_real_no_grad = f_real.detach()    # 创建一个新的张量，与原张量共享数据，不会在自动梯度计算中跟踪其历史，减少内存消耗
#         loss = self.criterion(f_fake, f_real)
#         return loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        model = models.vgg16(pretrained=True).features[:16]
        self.vgg = model.cuda()    # To ensure torch.cuda.FloatTensor
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = nn.functional.l1_loss(x_vgg, y_vgg)
        return loss