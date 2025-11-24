ReadMe -- 不同数据集和网络模型的修改

1）不同数据集类型

在 VM_UNet/configs/config_setting.py  ==第 43行选择数据集类型==

① 下载的内窥镜数据集：'polyp'   【路径：在第35行更改】

② 采集的膝关节软骨数据集：'cartilage'   **【路径：在第49行更改】**



2）不同数据集的mask处理不同

在 VM_UNet/datasets/dataset.py 第40-45行之间定义==msk的计算==

① 下载的内窥镜数据集polyp：mask的目标区域是白色（mask∈[0,255]） 

```python
msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
```

② 手机采集的软骨数据集（CartilageData_202312+202401）：mask的目标区域是黑色（mask∈[0,1]） 

```python
msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2)
```

③ 内窥镜采集的软骨数据集（CartilageData_endoscope）：mask的目标区域是红色（mask∈[0,38]） 【labelme转化得到的】

```python
msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 38
```

④ 所有整合的软骨数据集(CartilageData_all)：mask的目标区域是白色（mask∈[0,255]） 

```python
msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
```

3）不同注意力机制的VMUNet模型

在 train.py 的第6 / 7 行：import 不同的模型定义



4） 不同的损失函数

在 VM_UNet/configs/config_setting.py  ==第 62-65 行选择损失函数类型==

```python
criterion = BceDiceLoss(wb=1, wd=1)    # 二分类交叉熵Dice损失
criterion = HausdorffDiceLoss(alpha=0.3, wb=1, wd=1)  # Hausdorff距离损失+二分类交叉熵Dice损失
criterion = Perception_BceDiceLoss(wbce=1, wdice=1, wper=1)    # , alpha=0.8
criterion = SSIMBceDiceLoss(wbce=1, wdice=1, wssim=1)   # wssim=0.5
```



5）不同的预训练模型的加载方式不同：

在 /home/data/glw/Projects/VM_UNet/models/vmunet/vmunet.py   或 vmunet_STN.py的第81和95行：

对于vssm_small_0229_ckpt_epoch_222.pth ： 使用 pretrained_dict = modelCheckpoint['model']，‘model’中存储有模型的参数；

对于vssmsmall_upernet_4xb4-160k_ade20k-640x640_iter_112000.pth ： 使用 pretrained_dict = modelCheckpoint['state_dict']；

对于latest.pth,即我自己在VMUNet上训练得到的、存储在result/中的latest.pth：使用 pretrained_dict = modelCheckpoint['model_state_dict']；(在训练过程中的写入操作中仅保存了model_state_dict的网络相关参数)

6）对于Train_Unet【UNet的训练】

需要对  /data/glw/Projects/VM_UNet/engine.py 中 train_one_epoch （test / val one epoch）进行一条语句的增加，即

```python
out = torch.sigmoid(out)    # todo: For train_unet, normalize output to [0,1] ---- added by glw
```

是为了解决报错：/aten/src/ATen/native/cuda/Loss.cu:94: operator(): block: [300,0,0], thread: [96,0,0] Assertion `input_val >= zero && input_val <= one` failed.

> 错误提示“Assertion `input_val >= zero && input_val <= one` failed”通常发生在使用PyTorch进行深度学习训练时，尤其是在计算损失函数时。这个断言错误表明损失函数的输入值不在预期的[0, 1]范围内。以下是一些可能的原因和解决方案：
>
> 1. **数据类型不匹配**：确保所有输入张量的数据类型与模型预期的类型一致。在代码中可以使用 `.float()` 或 `.long()` 方法进行类型转换。
> 2. **验证索引范围**：在进行索引操作之前，确保索引值在张量的有效范围内。对于分类任务，确保目标标签的索引值在类别数的范围内。
> 3. **初始化张量**：在使用张量之前，确保所有张量已正确初始化。
> 4. **检查损失计算**：如果使用的是二元交叉熵损失（BCELoss）或类似需要预测值在[0, 1]范围内的损失函数，**确保模型的输出经过了sigmoid激活函数，以确保输出值在[0, 1]之间。**
> 5. **检查数据是否有NaN值**：计算损失时出现NaN值可能导致其取值范围过大或过小，即`input_val >= zero && input_val <= one` failed。检查输入数据是否有NaN，并将其替换为指定值。
> 6. **使用调试工具**：使用如NVIDIA Nsight等调试工具可以帮助捕获并定位问题所在。
> 7. **逐步检查代码**：如果错误不是在网络训练一开始就发生，而是在训练过程中突然报错，需要逐步检查代码，特别是模型的输出层和损失函数的输入。



6) 尝试了另一个Mamba+UNet的工作,模型定义写在了../VM_UNet/models/vmunet/Mamba_UNet.py
在 VM_UNet/configs/config_setting.py 第12行设置所使用的mamba网络类型，将在train.py的第78-90行调用不同的model进行训练。

7) 【运行结果不够好，弃用】若要使用advchain进行数据增强（https://github.com/cherise215/advchain）：
在 VM_UNet/engine.py 基础上进行改写了一个新的执行文件VM_UNet/engine_advchain.py:从45--143行是新增的图像变换及其损失定义的模块

运用：将train.py的第10行改为第11行，调用engine_advchain.py进行train_one_epoch

8) MedAugment: 用于图像分类和分割的自动数据增强插件
参考：https://zhuanlan.zhihu.com/p/642459888
在 VM_UNet/configs/config_setting.py 基础上进行改写了一个新的执行文件VM_UNet/configs/config_setting_MedAugment.py:从117--156行是利用albumentations库改写的图像变换的模块

运用：将train.py的第16行改为第17-18行，调用configs.config_setting_MedAugment.py进行config的配置

9) 放弃（8）的做法【即，放弃使用config_setting_MedAugment.py】，
看了Medaugment的原代码，是先用Medaugment做 data generation, 新生成的4倍数据集再带入到网络中去训练(4倍，即4次连续变换即可:过多的连续操作可能生成与原始图像差距较大的图像)
重新爬取了源码，运行 /home/data/glw/Projects/VM_UNet/MedAugment-master/utils/generation.py【注意此时要保证图像的长宽是一致的，因为存在旋转0度的操作】
（可使用/home/data/glw/Projects/CartilageData_all/image_resize.py对文件夹下的图片大小进行批量处理）

得到的训练集大小为4165张的新的数据集（/home/data/glw/Projects/VM_UNet/data_medaugment）
【注意】在train.py中注释掉from configs.config_setting_MedAugment import setting_config, MedAugment = False不再使用

在config_setting.py的第50行：更改数据集路径

10) 修改激活函数：
VM-UNet的核心模块是来自VMamba的VSS块，如上图所示。输入经过层归一化后，分成两个分支：第一个分支经过线性层和激活函数处理，第二个分支经过线性层、深度可分离卷积和激活函数处理，然后进入2D-Selective-Scan（SS2D）中进行处理。
在VM_UNet/models/vmunet/vmamba.py第302行和480行，SS2D模块定义中默认使用的是Silu函数（参考：https://www.cnblogs.com/kobeis007/p/16301398.html）
修改为Swish激活函数：由于Swish函数没有被内部封装，在vmamba.py第28-34行定义Swish函数，并在第302行进行引用：self.act = Swish()
（激活函数整理参考：https://zhuanlan.zhihu.com/p/92412922)


