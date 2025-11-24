import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs,save_imgs_visualize

def check_gradients(model, grad_clip=1):
    """
    检查模型梯度是否包含异常值（NaN或无穷大）。

    参数:
    model -- 要检查的PyTorch模型。
    grad_clip -- 梯度裁剪的阈值，超过该阈值的梯度将被视为异常。

    返回:
    bool -- 如果存在异常梯度，则返回True，否则返回False。
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 检查梯度是否包含NaN
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradient of {name}")
                return True
            # 检查梯度是否包含无穷大
            if torch.isinf(param.grad).any():
                print(f"Inf detected in gradient of {name}")
                return True
            # 检查梯度是否超出裁剪阈值
            if torch.any(torch.abs(param.grad) > grad_clip):
                print(f"Gradient norm too high in {name}, clipping...")
                with torch.no_grad():
                    param.grad.clamp_(-grad_clip, grad_clip)
    return False

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
        # non_blocking=True 作为 cuda() 函数的参数，数据迁移操作就会变成异步的，即数据开始被复制到 GPU 后，主机不需要等待它们全部被加载到 GPU 上，就可以继续执行其它操作
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        if torch.max(out)>1 or torch.min(out)<0:  # todo 若直接求Dice loss，会报错：Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
            max_value = torch.max(out)
            min_value = torch.min(out)
            # 应用最大最小归一化公式
            normalized_out = (out - min_value) / (max_value - min_value)
            out = normalized_out
        # out = torch.sigmoid(out)   # todo: For train_unet, normalize output to [0,1] ---- added by glw
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        # # todo:检查梯度是否存在异常值
        # if check_gradients(model):
        #     print("Training stopped due to gradient异常.")
        # else:
        #     print("Gradients are normal, training can continue.")
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

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
            if torch.max(out) > 1 or torch.min(out) < 0:  # todo 若直接求Dice loss，会报错：Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
                max_value = torch.max(out)
                min_value = torch.min(out)
                # 应用最大最小归一化公式
                normalized_out = (out - min_value) / (max_value - min_value)
                out = normalized_out
            # out = torch.sigmoid(out)    # todo: For train_unet, normalize output to [0,1] ---- added by glw
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
            # out = torch.sigmoid(out)    # todo: For train_unet, normalize output to [0,1] ---- added by glw
            if torch.max(out) > 1 or torch.min(out) < 0:  # todo 若直接求Dice loss，会报错：Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
                max_value = torch.max(out)
                min_value = torch.min(out)
                # 应用最大最小归一化公式
                normalized_out = (out - min_value) / (max_value - min_value)
                out = normalized_out
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
                # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
                # todo by glw
                save_imgs_visualize(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

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