import os
import sys
import time
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# 导入模型和工具函数
from models.vmunet.vmunet_XLSTM import XLSTM_VMUNet
from utils import myNormalize, myToTensor, myResize

# ==================== 配置参数 ====================
# 模型权重文件路径
WEIGHT_PATH = '/home/csd/mamba/VM_UNet/results/xlstm_vmunet_cartilage_pig_Wednesday_12_November_2025_12h_34m_31s/checkpoints/best-epoch196-loss0.2149.pth'

# 输入图片路径
IMAGE_PATH = '/home/csd/mamba/VM_UNet/test_picture/picture/0333.png'

# 输出路径（如果为None，则自动生成）
OUTPUT_PATH = '/home/csd/mamba/VM_UNet/test_picture/output'  # 例如: 'output/prediction.png'

# 设备设置
DEVICE = 'cuda'  # 'cuda' 或 'cpu'

# 数据集名称（用于归一化）
DATASET_NAME = 'cartilage_pig'  # 'cartilage_pig', 'cartilage', 'polyp', 'isic18', 'isic17' 等

# 输入图片尺寸
INPUT_SIZE = (256, 256)  # (height, width)

# 二值化阈值
THRESHOLD = 0.5

# ==================== 配置参数结束 ====================


def load_model(weight_path, device='cuda'):
    """
    加载训练好的模型权重
    
    Args:
        weight_path: 权重文件路径 (.pth文件)
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 加载好权重的模型
    """
    # 模型配置参数（需要与训练时保持一致）
    model_cfg = {
        'num_classes': 1,
        'input_channels': 3,
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 9, 2, 2],
        'drop_path_rate': 0.2,
        'load_ckpt_path': None,  # 推理时不需要预训练权重
        'xLSTM_layers': ['s', 'm'],  # xLSTM层配置
        'xLSTM_input_size': 256,
        'xLSTM_hidden_size': 64,
        'xLSTM_num_heads': 1,
    }
    
    # 初始化模型
    model = XLSTM_VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
        xLSTM_layers=model_cfg['xLSTM_layers'],
        xLSTM_input_size=model_cfg['xLSTM_input_size'],
        xLSTM_hidden_size=model_cfg['xLSTM_hidden_size'],
        xLSTM_num_heads=model_cfg['xLSTM_num_heads'],
    )
    
    # 加载权重
    print(f"Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    
    # 检查checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # 如果是训练时保存的完整checkpoint
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            # 如果是其他格式
            state_dict = checkpoint['model']
        else:
            # 如果直接是state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, dataset_name='cartilage_pig', input_size=(256, 256)):
    """
    预处理输入图片
    
    Args:
        image_path: 图片路径
        dataset_name: 数据集名称（用于归一化）
        input_size: 输入尺寸 (height, width)
    
    Returns:
        img_tensor: 预处理后的图片张量 [1, 3, H, W]
        original_img: 原始图片（用于可视化）
    """
    # 读取图片
    if isinstance(image_path, str):
        img = np.array(Image.open(image_path).convert('RGB'))
    else:
        img = image_path
    
    original_img = img.copy()
    
    # 应用预处理（与训练时保持一致）
    # 注意：这里使用test_transformer的配置
    transformer = transforms.Compose([
        myNormalize(dataset_name, train=False),
        myToTensor(),
        myResize(input_size[0], input_size[1])
    ])
    
    # 创建dummy mask用于transformer（transformer需要img和mask）
    dummy_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    
    # 应用transformer
    img_tensor, _ = transformer((img, dummy_mask))
    
    # 确保数据类型为 float32（模型要求）
    if img_tensor.dtype != torch.float32:
        img_tensor = img_tensor.float()
    
    # 添加batch维度
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, original_img


def postprocess_output(output, input_size=(256, 256), threshold=0.5):
    """
    后处理模型输出
    
    Args:
        output: 模型输出 [B, H, W*num_classes] 或 [B, H, W, num_classes]
        input_size: 输入图像尺寸 (height, width)
        threshold: 二值化阈值
    
    Returns:
        prediction: 二值化后的预测结果 [H, W]
        probability: 概率图 [H, W]
    """
    # 转换为numpy
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    
    # 处理不同的输出形状
    if len(output.shape) == 3:
        # [B, seq_len, channels] - xLSTM的输出格式
        batch_size, seq_len, channels = output.shape
        height, width = input_size
        
        # 根据模型forward逻辑：
        # logits从vmunet输出 [B, num_classes, H, W]
        # 经过permute和view后变成 [B, H, W*num_classes]
        # xLSTM处理后仍然是 [B, H, W*num_classes] 或 [B, H, W] (如果num_classes=1)
        
        if channels == width:
            # num_classes=1的情况，直接reshape [B, H, W]
            output = output.reshape(batch_size, height, width)
        elif channels % width == 0:
            # 多类别情况 [B, H, W*num_classes]
            num_classes = channels // width
            output = output.reshape(batch_size, height, width, num_classes)
            # 对于二分类，取第一个通道或使用argmax
            if num_classes == 1:
                output = output.squeeze(-1)
            else:
                # 多类别：取第一个通道（或者可以根据需要改为argmax）
                output = output[:, :, :, 0]
        else:
            # 如果无法整除，尝试直接reshape为 [B, H, W]
            # 这可能发生在某些特殊配置下
            if seq_len == height:
                output = output.reshape(batch_size, height, width)
            else:
                raise ValueError(f"Cannot reshape output with shape {output.shape} to image shape ({height}, {width})")
    
    elif len(output.shape) == 4:
        # [B, num_classes, H, W] 或 [B, H, W, num_classes]
        if output.shape[1] == 1 or output.shape[1] < output.shape[2]:
            # [B, 1, H, W] 或 [B, num_classes, H, W]
            output = output.squeeze(1)  # [B, H, W]
        elif output.shape[-1] == 1:
            # [B, H, W, 1]
            output = output.squeeze(-1)  # [B, H, W]
    
    # 移除batch维度
    if len(output.shape) == 3:
        output = output.squeeze(0)  # [H, W]
    
    # 如果输出还没有经过sigmoid，检查是否需要应用sigmoid
    # 注意：模型已经应用了sigmoid（num_classes=1时），但为了安全起见还是检查一下
    if output.max() > 1.0 or output.min() < 0.0:
        # 如果值域不在[0,1]，应用sigmoid
        output = 1 / (1 + np.exp(-np.clip(output, -500, 500)))  # 防止溢出
    
    # 确保值域在[0,1]
    output = np.clip(output, 0, 1)
    
    # 获取概率图
    probability = output
    
    # 二值化
    prediction = (probability >= threshold).astype(np.uint8) * 255
    
    return prediction, probability


def inference_single_image(model, image_path, device='cuda', 
                           dataset_name='cartilage_pig', 
                           input_size=(256, 256),
                           threshold=0.5,
                           save_path=None):
    """
    对单张图片进行推理
    
    Args:
        model: 加载好的模型
        image_path: 图片路径
        device: 设备
        dataset_name: 数据集名称
        input_size: 输入尺寸
        threshold: 二值化阈值
        save_path: 保存路径（可选）
    
    Returns:
        prediction: 二值化预测结果
        probability: 概率图
    """
    # 记录总开始时间
    total_start_time = time.time()
    
    # 预处理
    preprocess_start = time.time()
    img_tensor, original_img = preprocess_image(image_path, dataset_name, input_size)
    
    # 确保数据类型为 float32，然后移动到设备
    if img_tensor.dtype != torch.float32:
        img_tensor = img_tensor.float()
    img_tensor = img_tensor.to(device)
    preprocess_time = time.time() - preprocess_start
    
    # 推理（纯模型前向传播时间）
    print("Running inference...")
    # 预热（如果使用GPU）
    if device == 'cuda':
        with torch.no_grad():
            _ = model(img_tensor)
        torch.cuda.synchronize()  # 等待GPU完成
    
    # 正式推理
    inference_start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    if device == 'cuda':
        torch.cuda.synchronize()  # 等待GPU完成
    inference_time = time.time() - inference_start
    
    # 后处理
    postprocess_start = time.time()
    prediction, probability = postprocess_output(output, input_size, threshold)
    
    # 如果原始图片尺寸与输入尺寸不同，需要resize回原始尺寸
    if original_img.shape[:2] != input_size:
        prediction = cv2.resize(prediction, (original_img.shape[1], original_img.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        probability = cv2.resize(probability, (original_img.shape[1], original_img.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
    postprocess_time = time.time() - postprocess_start
    
    # 总时间
    total_time = time.time() - total_start_time
    
    # 保存结果
    save_start = time.time()
    if save_path is not None:
        # 保存二值化结果
        cv2.imwrite(save_path, prediction)
        print(f"Prediction saved to: {save_path}")
        
        # 保存概率图
        prob_path = save_path.replace('.png', '_probability.png')
        prob_img = (probability * 255).astype(np.uint8)
        cv2.imwrite(prob_path, prob_img)
        print(f"Probability map saved to: {prob_path}")
        
        # 保存可视化结果（原图+预测叠加）
        vis_path = save_path.replace('.png', '_visualization.png')
        vis_img = original_img.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        
        # 创建彩色mask（红色）
        colored_mask = np.zeros_like(vis_img)
        colored_mask[prediction > 0] = [255, 0, 0]  # 红色
        
        # 叠加显示
        overlay = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(vis_path, overlay)
        print(f"Visualization saved to: {vis_path}")
    save_time = time.time() - save_start
    
    # 打印时间统计
    print("\n" + "=" * 50)
    print("Time Statistics:")
    print("=" * 50)
    print(f"Preprocessing time:  {preprocess_time*1000:.2f} ms")
    print(f"Inference time:      {inference_time*1000:.2f} ms")
    print(f"Postprocessing time: {postprocess_time*1000:.2f} ms")
    print(f"Save time:           {save_time*1000:.2f} ms")
    print(f"Total time:          {total_time*1000:.2f} ms")
    print(f"FPS:                 {1.0/total_time:.2f} frames/second")
    print("=" * 50)
    
    return prediction, probability


def main():
    # 检查设备
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = DEVICE
    
    # 设置输出路径
    if OUTPUT_PATH is None:
        # 如果未指定输出路径，在输入图片同目录下生成
        base_name = os.path.splitext(IMAGE_PATH)[0]
        output_path = f"{base_name}_pred.png"
    else:
        # 检查 OUTPUT_PATH 是目录还是文件路径
        if os.path.isdir(OUTPUT_PATH):
            # 如果是目录，自动生成文件名
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            input_filename = os.path.basename(IMAGE_PATH)
            base_name = os.path.splitext(input_filename)[0]
            output_path = os.path.join(OUTPUT_PATH, f"{base_name}_pred.png")
        elif os.path.isfile(OUTPUT_PATH) or OUTPUT_PATH.endswith(('.png', '.jpg', '.jpeg')):
            # 如果是文件路径，直接使用
            output_path = OUTPUT_PATH
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        else:
            # 可能是新文件路径，检查目录是否存在
            output_dir = os.path.dirname(OUTPUT_PATH)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_path = OUTPUT_PATH
    
    # 检查权重文件是否存在
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"Model weight file not found: {WEIGHT_PATH}")
    
    # 检查输入图片是否存在
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Input image file not found: {IMAGE_PATH}")
    
    print("=" * 50)
    print("XLSTM_VMUNet Inference")
    print("=" * 50)
    print(f"Weight path: {WEIGHT_PATH}")
    print(f"Image path: {IMAGE_PATH}")
    print(f"Output path: {output_path}")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Threshold: {THRESHOLD}")
    print("=" * 50)
    
    # 加载模型
    model = load_model(WEIGHT_PATH, device=device)
    
    # 推理
    prediction, probability = inference_single_image(
        model=model,
        image_path=IMAGE_PATH,
        device=device,
        dataset_name=DATASET_NAME,
        input_size=INPUT_SIZE,
        threshold=THRESHOLD,
        save_path=output_path
    )
    
    print("\n" + "=" * 50)
    print("Inference completed!")
    print("=" * 50)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction value range: [{prediction.min()}, {prediction.max()}]")
    print(f"Probability value range: [{probability.min():.4f}, {probability.max():.4f}]")
    print("=" * 50)


if __name__ == '__main__':
    main()

