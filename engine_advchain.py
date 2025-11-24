import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs

from advchain.advchain.augmentor.adv_bias import AdvBias
from advchain.advchain.augmentor.adv_morph import AdvMorph
from advchain.advchain.augmentor.adv_noise import AdvNoise
from advchain.advchain.augmentor.adv_affine import AdvAffine
from advchain.advchain.augmentor.adv_compose_solver import ComposeAdversarialTransformSolver

from advchain.advchain.common.utils import random_chain,load_image_label
from advchain.advchain.common.loss import cross_entropy_2D


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images_tensor, targets_tensor = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        init_output = model(images_tensor)

        ## advain_augment
        bs = config.batch_size     # batch_size
        im_ch = 3  # 1
        crop_size = [256, 256]
        augmentor_bias = AdvBias(
            config_dict={'epsilon': 0.3,
                         'control_point_spacing': [128, 128],
                         'downscale': 4,  # 3    # WARNING:root:control point spacing may be too large, please increase the downscale factor.
                         'data_size': (bs, im_ch, crop_size[0], crop_size[1]),
                         'interpolation_order': 3,
                         'init_mode': 'random',
                         'space': 'log'}, debug=False)

        augmentor_noise = AdvNoise(config_dict={'epsilon': 1,
                                                'xi': 1e-6,
                                                'data_size': (bs, im_ch, crop_size[0], crop_size[1])},
                                   debug=False)

        augmentor_affine = AdvAffine(config_dict={
            'rot': 30 / 180,
            'scale_x': 0.2,
            'scale_y': 0.2,
            'shift_x': 0.1,
            'shift_y': 0.1,
            'shear_x': 0.,
            'shear_y': 0.,
            'data_size': (bs, im_ch, crop_size[0], crop_size[1]),
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'},
            debug=False)

        augmentor_morph = AdvMorph(
            config_dict=
            {'epsilon': 1.5,
             'data_size': (bs, im_ch, crop_size[0], crop_size[1]),
             'vector_size': [crop_size[0] // 16, crop_size[1] // 16],
             'forward_interp': 'bilinear',
             'backward_interp': 'bilinear'},
            debug=False)

        # ## keep model fixed, set up a solver
        # model.eval()
        ## specify the transformation chain
        step_sizes = [1, 1, 1, 1]
        transformation_chain = [augmentor_noise, augmentor_bias, augmentor_morph, augmentor_affine]
        one_chain = random_chain(transformation_chain.copy(), max_length=len(transformation_chain))  # 随机选取transform_chain中一定的长度，将进行trans转换

        solver = ComposeAdversarialTransformSolver(
            chain_of_transforms=one_chain,  # transformation_chain
            divergence_types=['mse', 'contour'],  ### you can also change it to 'kl'.
            divergence_weights=[1.0, 0.5],
            use_gpu=True,
            debug=True,
            if_norm_image=True,
            is_gt=False
        )
        ## 4
        # random initialization
        # solver.init_random_transformation()
        # rand_transformed_image = solver.forward(images_tensor.detach().clone())
        # rand_predict = model.forward(rand_transformed_image)
        #  # for visualization
        # warp_back_rand_predict = solver.predict_backward(rand_predict)
        #
        # rand_bias = augmentor_bias.bias_field
        # rand_noise = augmentor_noise.param
        # rand_dxy, rand_morph = augmentor_morph.get_deformation_displacement_field(-augmentor_morph.param)

        ## compute consistency loss (adversarial loss)
        reg_loss = solver.adversarial_training(     # is_gt=True: 6.0557e-07
            data=images_tensor, model=model, init_output=init_output.detach().clone(),
            n_iter=1,
            lazy_load=[False] * len(one_chain),    # len(one_chain)
            optimize_flags=[True] * len(one_chain),    # len(one_chain)
            ## you can also turn off adversarial training for one particular transformation
            step_sizes=[1] * len(one_chain))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 0.00001
        optimizer.zero_grad(set_to_none=True)
        model.train()
        model.zero_grad()
        init_output = model(images_tensor)

        ## compute supervised loss
        # supervised_loss = cross_entropy_2D(init_output, targets_tensor)
        supervised_loss = criterion(init_output,targets_tensor) # todo:glw
        lamda = 1
        total_loss = supervised_loss + lamda * (-1)*reg_loss     # (-1)*reg_loss 最大化分歧??
        total_loss.backward()
        optimizer.step()
        ## !important, please reset the transformation parameters.
        solver.reset_transformation()

        # out = model(images)
        # loss = criterion(out, targets)
        # loss.backward()
        # optimizer.step()
        
        loss_list.append(total_loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', total_loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:        # config.val_interval=30
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()   # pred images
            # cv2.imwrite(out,'/home/data/glw/Projects/PraNet/results/VMUNet_cartilagescope/cartilage/{}.png'.format(test_data_name))
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs_s/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)