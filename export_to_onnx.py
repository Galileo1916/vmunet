import os
import sys
import torch
import torch.onnx
import numpy as np

# 导入模型
from models.vmunet.vmunet_XLSTM import XLSTM_VMUNet

# ==================== 配置参数 ====================
# 模型权重文件路径
WEIGHT_PATH = '/home/csd/mamba/VM_UNet/results/xlstm_vmunet_cartilage_pig_Wednesday_12_November_2025_12h_34m_31s/checkpoints/best-epoch196-loss0.2149.pth'

# ONNX输出路径
ONNX_OUTPUT_PATH = '/home/csd/mamba/VM_UNet/results/xlstm_vmunet_cartilage_pig_Wednesday_12_November_2025_12h_34m_31s/checkpoints/best-epoch196-loss0.2149.onnx'

# 设备设置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 输入图片尺寸 (height, width)
INPUT_SIZE = (256, 256)

# 批次大小
BATCH_SIZE = 1

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


def export_to_onnx(model, output_path, input_size=(256, 256), batch_size=1, device='cuda'):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model: 加载好的PyTorch模型
        output_path: ONNX文件保存路径
        input_size: 输入图像尺寸 (height, width)
        batch_size: 批次大小
        device: 设备
    """
    print("\n" + "=" * 50)
    print("Exporting model to ONNX format...")
    print("=" * 50)
    
    # 创建示例输入
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1]).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 设置输入和输出名称
    input_names = ['input_image']
    output_names = ['output_segmentation']
    
    # 动态轴配置（支持不同batch size）
    dynamic_axes = {
        'input_image': {0: 'batch_size'},
        'output_segmentation': {0: 'batch_size'}
    }
    
    print(f"Exporting to: {output_path}")
    
    # 尝试不同的opset版本（因为GroupNorm在不同版本中有不同的支持）
    opset_versions = [17, 16, 13, 11, 14]  # 按优先级排序
    export_success = False
    
    for opset_version in opset_versions:
        try:
            print(f"\nTrying opset version {opset_version}...")
            
            # 导出ONNX模型
            torch.onnx.export(
                model,                      # 模型
                dummy_input,                # 示例输入
                output_path,                # 输出路径
                export_params=True,         # 导出训练好的参数
                opset_version=opset_version, # ONNX opset版本
                do_constant_folding=False,   # 常量折叠优化
                input_names=input_names,    # 输入名称
                output_names=output_names,  # 输出名称
                dynamic_axes=dynamic_axes,  # 动态轴
                verbose=False,              # 不显示详细信息
            )
            
            print(f"✓ ONNX export completed successfully with opset version {opset_version}!")
            print(f"✓ Model saved to: {output_path}")
            export_success = True
            break
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed with opset version {opset_version}")
            print(f"  Error: {error_msg[:200]}")
            
            # 如果是最后一个版本，提供详细错误分析
            if opset_version == opset_versions[-1]:
                print("\n" + "=" * 50)
                print("错误分析 (Error Analysis):")
                print("=" * 50)
                
                if "list index out of range" in error_msg or "GroupNorm" in error_msg or "group_norm" in error_msg:
                    print("\n原因 (Root Cause):")
                    print("- 错误发生在导出 GroupNorm 操作时")
                    print("- ONNX转换器在处理GroupNorm的axes参数时遇到问题")
                    print("- 这是PyTorch ONNX导出器与某些opset版本的兼容性问题")
                    print("\n可能的解决方案 (Possible Solutions):")
                    print("1. 更新PyTorch版本: pip install --upgrade torch")
                    print("2. 更新ONNX版本: pip install --upgrade onnx")
                    print("3. 如果所有方法都失败，可能需要修改模型代码中的GroupNorm实现")
                
                print("\n其他可能的问题:")
                print("- xLSTM中的循环操作（for循环）可能不完全支持ONNX导出")
                print("- 模型中的某些PyTorch操作可能没有对应的ONNX算子")
                print("- 动态形状的处理可能需要特殊处理")
                
                raise
    
    if not export_success:
        raise RuntimeError("Failed to export ONNX model with all tested opset versions")
    
    # 验证导出的ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")
        
        # 打印模型信息
        print("\n" + "=" * 50)
        print("Model Information:")
        print("=" * 50)
        print(f"Input shape: {[batch_size, 3, input_size[0], input_size[1]]}")
        print(f"Input name: {input_names[0]}")
        print(f"Output name: {output_names[0]}")
        
        # 获取输出形状
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
        
    except ImportError:
        print("⚠ Warning: onnx package not found. Cannot verify the exported model.")
        print("  Install with: pip install onnx")
    except Exception as e:
        print(f"⚠ Warning: ONNX model verification failed: {e}")


def main():
    # 检查权重文件是否存在
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"Model weight file not found: {WEIGHT_PATH}")
    
    print("=" * 50)
    print("XLSTM_VMUNet ONNX Export")
    print("=" * 50)
    print(f"Weight path: {WEIGHT_PATH}")
    print(f"ONNX output path: {ONNX_OUTPUT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 50)
    
    # 加载模型
    model = load_model(WEIGHT_PATH, device=DEVICE)
    
    # 导出ONNX
    export_to_onnx(
        model=model,
        output_path=ONNX_OUTPUT_PATH,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    print("\n" + "=" * 50)
    print("Export completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()

