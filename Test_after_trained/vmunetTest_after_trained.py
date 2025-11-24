import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter

from models.vmunet.vmunet_Sobel import VMUNet_Sobel
# from models.vmunet.vmunet import VMUNet
from models.vmunet.Mamba_UNet import MambaUnet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config
# from configs.config_setting_MedAugment import setting_config, MedAugment = False

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
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
    model_cfg = config.model_config
    # todo: 先修改config_setting.py第12行  network 的类型
    if 'vmunet' in config.network:
        model = VMUNet_Sobel(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
        )
        model.load_from()
    elif config.network == 'mambaunet':
        model = MambaUnet(model_cfg)
        model.load_from(model_cfg)

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)



    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Testing----------#')
    # todo: for only test - the output path [与mmseg产生的预训练模型的结构不同，相当于mmseg产生的预训练模型中的stateDict]
    best_weight = torch.load(
        '/data/glw/Projects/VM_UNet/results/vmunet_CSAM_Sobel_cartilage_Friday_20_December_2024_06h_09m_51s/checkpoints/best.pth',
        map_location=torch.device('cpu'))

    model.load_state_dict(best_weight)
    loss = test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
    )



if __name__ == '__main__':
    config = setting_config

    main(config)