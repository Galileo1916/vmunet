import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.UNets.models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
from models.UNets.swin_transformer import SwinUnet
from models.UNets.vit_seg_modeling import VisionTransformer as TransUNet
from models.UNets.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from engine_unet import *
import os
import sys

from utils import *
from configs.config_setting_UNets import setting_config
# from configs.config_setting_MedAugment import setting_config, MedAugment = False

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs_s')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)  # MedAugment = MedAugment
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
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





    print('#----------Prepareing Model----------#')
    # model_cfg = config.model_config
    # todo: 先修改config_setting_UNets.py第12行  network 的类型
    if config.network == 'unet':
        model = U_Net()
        # model.load_from()
    elif config.network == 'rcnn_unet':
        model = R2U_Net()
    elif config.network == 'atten_unet':
        model = AttU_Net()
    elif config.network == 'atten_rcnn_unet':
        model = R2AttU_Net()
    elif config.network == 'nested_unet':
        model = NestedUNet()
    elif config.network == 'swin_unet':
        model_config = {
            'img_size':config.input_size_h,  # 'image_size':256,   # 224
            'num_classes': 1,
            'input_channels': 3,
            # 'load_ckpt_path': '/data/glw/Projects/SwinUNet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth',
            'load_ckpt_path':'/data/glw/Projects/VM_UNet/pre_trained_weights/swin_large_patch4_window7_224_22k.pth',
            'patch_size': 4,
            'EMBED_DIM': 96,
            'DEPTHS': [2, 2, 2, 2],
            'DECODER_DEPTHS': [2, 2, 2, 1],
            'NUM_HEADS': [3, 6, 12, 24],
            'WINDOW_SIZE': 8,  # 8   # 256 // 8 = 32( = unet channel)
            'MLP_RATIO': 4.,
            'drop_rate': 0.,
            'DROP_PATH_RATE': 0.2,
            'swin_patch_norm': True,
            'use_checkpoint': True,
            'QKV_BIAS': True,
            'QK_SCALE': None,
            'APE': False,
            'PATCH_NORM': True,
        }
        model = SwinUnet(model_config, num_classes=config.num_classes)
        model.load_from(model_config)
    elif config.network == 'transunet_R50':
        vit_name = 'R50-ViT-B_16'
        model_config = CONFIGS_ViT_seg[vit_name]
        model_config.vit_patches_size = 64     # todo  # default=8
        model_config.n_classes = config.num_classes
        model_config.n_skip = 3   # todo:default
        if config.network.find('R50') != -1:
            model_config.patches.grid = (   # 4
            int(config.input_size_h / model_config.vit_patches_size), int(config.input_size_h / model_config.vit_patches_size))
        model = TransUNet(model_config, img_size=config.input_size_h, num_classes=config.num_classes)
        model.load_from(weights=np.load(model_config.pretrained_path))
        
    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)



    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )



if __name__ == '__main__':
    config = setting_config

    main(config)