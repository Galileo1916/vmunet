"""
ONNX兼容版本的XLSTM_VMUNet模型
主要修改：将GroupNorm替换为ONNX兼容的归一化方式

使用方法：
1. 备份原始文件：cp vmunet_XLSTM.py vmunet_XLSTM_original.py
2. 复制这个文件：cp vmunet_XLSTM_onnx_compatible.py vmunet_XLSTM.py
3. 或者修改导入语句使用这个文件

归一化选项：
- 'layernorm': 使用LayerNorm（推荐，ONNX完全支持）
- 'identity': 不使用归一化（最简单，但可能影响性能）
- 'groupnorm': 使用原始GroupNorm（如果ONNX版本更新后支持）
"""

import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 归一化方式选择（用于ONNX导出兼容性）
# 选项: 'layernorm', 'identity', 'groupnorm'
NORMALIZATION_MODE = 'layernorm'  # 默认使用LayerNorm

# 复制其他所有类和函数...
# 这里我们只展示需要修改的部分

# ==================== 需要修改的类 ====================

class GroupNormONNXCompatible(nn.Module):
    """
    ONNX兼容的GroupNorm实现
    使用LayerNorm作为替代，因为LayerNorm在ONNX中完全支持
    """
    def __init__(self, num_groups, num_channels, mode='layernorm'):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.mode = mode
        
        if mode == 'layernorm':
            # 使用LayerNorm替代GroupNorm
            # LayerNorm对每个样本的所有通道进行归一化
            self.norm = nn.LayerNorm(num_channels)
        elif mode == 'identity':
            # 不使用归一化
            self.norm = nn.Identity()
        elif mode == 'groupnorm':
            # 使用原始GroupNorm（如果ONNX支持）
            self.norm = nn.GroupNorm(num_groups, num_channels)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
    
    def forward(self, x):
        """
        x: (batch_size, num_channels) 对于sLSTM和mLSTM的情况
        """
        if self.mode == 'layernorm':
            # LayerNorm期望输入为 (batch_size, num_channels)
            return self.norm(x)
        else:
            # Identity或GroupNorm
            return self.norm(x)


class sLSTMBlockONNXCompatible(nn.Module):
    """
    ONNX兼容版本的sLSTMBlock
    主要修改：使用LayerNorm替代GroupNorm
    """
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4 / 3, norm_mode='layernorm'):
        """
        Parameters:
        - norm_mode: 归一化方式 'layernorm', 'identity', 'groupnorm'
        """
        super(sLSTMBlockONNXCompatible, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor
        
        assert hidden_size % num_heads == 0
        assert proj_factor > 0
        
        # 导入BlockDiagonal和CausalConv1D（需要从原文件复制）
        # 这里假设它们已经在模块中定义
        
        # Layers
        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)
        
        # Block-diagonal weight matrices
        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)
        
        # Recurrent weight matrices
        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)
        
        # ONNX兼容的归一化层（替换GroupNorm）
        if norm_mode == 'layernorm':
            self.group_norm = nn.LayerNorm(hidden_size)
        elif norm_mode == 'identity':
            self.group_norm = nn.Identity()
        else:  # 'groupnorm'
            self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        
        # Projections
        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(hidden_size * proj_factor), input_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        
        # 确保tensor在正确的设备上
        device = x.device
        H_prev = h_prev.to(device)
        C_prev = c_prev.to(device)
        N_prev = n_prev.to(device)
        M_prev = m_prev.to(device)
        
        # Normalize input
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))
        
        # Compute gates
        z = torch.tanh(self.Wz(x) + self.Rz(H_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(H_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(H_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(H_prev)
        
        # Stable computation
        m_t = torch.max(f_tilde + M_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + M_prev - m_t)
        
        # Update states
        c_t = f * C_prev + i * z
        n_t = f * N_prev + i
        h_t = o * c_t / n_t
        
        # Apply ONNX兼容的归一化
        output = h_t
        output_norm = self.group_norm(output)
        
        # Project and gate
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        
        # Residual connection
        final_output = output + x
        
        return final_output, (h_t, c_t, n_t, m_t)


class mLSTMBlockONNXCompatible(nn.Module):
    """
    ONNX兼容版本的mLSTMBlock
    主要修改：使用LayerNorm替代GroupNorm
    """
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2, norm_mode='layernorm'):
        super(mLSTMBlockONNXCompatible, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor
        
        assert hidden_size % num_heads == 0
        assert proj_factor > 0
        
        # Layers
        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)
        
        # Block diagonal matrices
        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        
        # LSTM gates
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)
        
        # ONNX兼容的归一化层（替换GroupNorm）
        if norm_mode == 'layernorm':
            self.group_norm = nn.LayerNorm(hidden_size)
        elif norm_mode == 'identity':
            self.group_norm = nn.Identity()
        else:  # 'groupnorm'
            self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        
        device = x.device
        H_prev = h_prev.to(device)
        C_prev = c_prev.to(device)
        N_prev = n_prev.to(device)
        M_prev = m_prev.to(device)
        
        # Normalize and project
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)
        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)
        
        # Attention
        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)
        
        # Gates
        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))
        
        # Memory update
        m_t = torch.max(f_tilde + M_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + M_prev - m_t)
        
        # Cell and hidden state
        c_t = f * C_prev + i * (v * k)
        n_t = f * N_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0]
        
        # Apply ONNX兼容的归一化
        output_norm = self.group_norm(h_t)
        output = output_norm + x_skip
        
        # Final projection
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x
        
        return final_output, (h_t, c_t, n_t, m_t)


# ==================== 使用说明 ====================
"""
这个文件提供了ONNX兼容的GroupNorm替换方案。

完整替换步骤：
1. 备份原始文件
2. 复制原文件中的所有其他类和函数（SS2D, VSSBlock, VSSM, xLSTM等）
3. 修改sLSTMBlock和mLSTMBlock类，将GroupNorm替换为LayerNorm
4. 或者直接修改原文件中的GroupNorm定义

快速修改方法（在原文件中）：
在vmunet_XLSTM.py中找到：
    self.group_norm = nn.GroupNorm(num_heads, hidden_size)
替换为：
    # ONNX兼容版本：使用LayerNorm替代GroupNorm
    self.group_norm = nn.LayerNorm(hidden_size)
    # 如果仍想使用GroupNorm，可以保留原代码并尝试更新PyTorch版本
"""

