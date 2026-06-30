"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm."""
        from einops import einsum  # 確保有引入
        import numpy as np         # 用於方便地輸出純文字檔案
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)  # 確保輸出檔案在當前程式碼所在的目錄下
        
        (b, l, d_in) = u.shape
        n = A.shape[1]
        def hardware_exp_approx(x_int, scale=256):
            """
            實作 MARCA 論文中的 e^x 近似硬體演算法 (純整數與位元運算)
            x_int: 已經放大 scale 倍的定點數張量 (必須為 32-bit 整數)
            scale: 放大倍率 (256 代表保留 8-bit 小數)
            """
            # 確保是整數型態，才能做位元運算
            x_int = x_int.to(torch.int32)
            
            # 1. 乘以 log2(e) ≒ 1.442695
            # 在定點數表示中，1.442695 * 256 ≒ 369
            LOG2E_INT = 369
            
            # x_int 乘以 369 後，小數點被放大了兩次 (256 * 256)
            # 所以我們要 >> 8 (除以 256) 退回一次，維持 8-bit 小數精度
            x_prime_int = torch.bitwise_right_shift(x_int * LOG2E_INT, 8)
            
            # 2. 提取整數 Z 與小數 f
            # 向右位移 8 bit 抓出整數 (PyTorch 的 >> 會自動處理 2 的補數符號延伸)
            Z = torch.bitwise_right_shift(x_prime_int, 8)  
            # 用 Bitwise AND ( & 0xFF ) 抓出低 8 bit 作為小數部分
            f_int = torch.bitwise_and(x_prime_int, 255)    
            
            # 3. 計算 2^f ≒ 1 + f 
            # 在 8-bit 定點數中，真實世界的數值 1.0，在暫存器裡的值就是 256
            two_to_f_int = 256 + f_int
            
            # 4. 乘以 2^Z
            # Z 是負數，乘以 2^Z 等同於向右位移 |Z|
            shift_amount = torch.abs(Z)
            
            # (安全機制) 避免位移量超過 31 導致 PyTorch 報錯，最大限制在 31
            # 在硬體中如果位移超過暫存器寬度，值會直接變成 0，這裡的 clamp 也能達到接近 0 的效果
            shift_amount = torch.clamp(shift_amount, max=31)
            
            # 執行最終的右移
            y_int = torch.bitwise_right_shift(two_to_f_int, shift_amount)
            
            # 轉回 float32 讓 PyTorch 能繼續做後續的 einsum
            return y_int.to(torch.float32)
        # 輔助函數：將 Tensor 匯出為純文字檔 (Testbench 格式)
        # 輔助函數：將 Tensor 匯出為純文字檔 (Testbench 格式)
        def export_tensor_to_txt(tensor, filename, is_int=False, is_hex=False, bit_width=8):
            # 將 tensor 轉為 numpy，並確保它在 CPU 上
            np_arr = tensor.detach().cpu().numpy()
            
            # 將多維陣列攤平成一維
            flat_arr = np_arr.flatten()
            
            if is_hex:
                # 處理 16 進位與 2 的補數轉換
                mask = (1 << bit_width) - 1
                hex_digits = bit_width // 4
                with open(filename, 'w') as f:
                    for val in flat_arr:
                        hex_val = int(val) & mask
                        f.write(f"{hex_val:0{hex_digits}X}\n")
            else:
                # 處理傳統的 10 進位整數或浮點數
                if is_int:
                    flat_arr = flat_arr.astype(np.int32)
                fmt = '%d' if is_int else '%.6f'
                np.savetxt(filename, flat_arr, fmt=fmt)
            # print(f"[硬體除錯] 已匯出 {tensor.shape} 的資料至 {filename}")

        # ======================================================================
        # 硬體模擬：將 A 與 B 的內部數值全部轉換為整數（捨棄小數點）
        # ======================================================================
        #export_tensor_to_txt(delta, "delta_float.txt", is_int=False)
        # A 方案一：直接無條件四捨五入轉整數
        A_int_tensor = torch.round(A).to(torch.int32)
        A_int = A_int_tensor.to(torch.float32)
        
        # B 方案二：硬體定點數量化 (Fixed-point Quantization)
        BIT_WIDTH_SCALE = 256  
        B_int_tensor = torch.round(B * BIT_WIDTH_SCALE).to(torch.int32)
        
        # 匯出 Testbench 輸入資料 (A 與 B 放大後的整數值)
        # B 是我們量化過後的整數，所以用 is_int=True 匯出乾淨的整數格式
        # 匯出 Testbench 輸入資料 (A 與 B 放大後的整數值)
        # 加入 is_hex=True 並設定硬體記憶體的位元寬度 (預設為 8-bit)
        # 注意：如果你的硬體對應暫存器是 16 或 32 bit，請把 bit_width 改成 16 或 32
        export_tensor_to_txt(A_int_tensor, "A_testbench.txt", is_hex=True, bit_width=16)
        export_tensor_to_txt(B_int_tensor, "B_testbench.txt", is_hex=True, bit_width=16)
        
        # 如果需要，你也可以匯出 delta 和 u 作為 testbench 輸入
        delta_int_tensor = torch.round(delta * BIT_WIDTH_SCALE).to(torch.int32)
        export_tensor_to_txt(delta_int_tensor, "delta_testbench.txt", is_hex=True, bit_width=16)
        delta_float_tensor = delta_int_tensor.to(torch.float32)
        B_int = B_int_tensor.to(torch.float32)
        # ======================================================================
        
        # 離散化連續參數
        deltaAnonE = einsum(delta_float_tensor, A_int, 'b l d_in, d_in n -> b l d_in n')
        deltaAnonE_float = einsum(delta, A_int, 'b l d_in, d_in n -> b l d_in n')
        
        # [修改] 中間乘積結果，建議使用 32-bit 匯出 (對應硬體的 Accumulator)
        export_tensor_to_txt(deltaAnonE, "deltaA_nonE.txt", is_hex=True, bit_width=32)
        
        deltaA_int = hardware_exp_approx(deltaAnonE, scale=BIT_WIDTH_SCALE)
        deltaA_float = torch.exp(deltaAnonE_float)
        
        deltaB = einsum(delta_float_tensor, B_int, 'b l d_in, b l n -> b l d_in n') 
        
        # 匯出硬體運算的標準答案 (Golden Answer)
        
        # [保留] 浮點數版本通常是給軟體驗證演算法 (Algorithm Debug) 用的，不轉 Hex
        export_tensor_to_txt(deltaA_float, "A_answer_exp_float.txt", is_int=False)
        
        # [修改] 整數標準答案轉為 Hex (同樣建議用 32-bit 對齊硬體暫存器寬度)
        export_tensor_to_txt(deltaA_int, "A_answer_exp_int.txt", is_hex=True, bit_width=32)
        export_tensor_to_txt(deltaB, "B_answer.txt", is_hex=True, bit_width=32)
        
        deltaB_u_Quantization = torch.round(deltaB / BIT_WIDTH_SCALE)
        deltaA = deltaA_int / BIT_WIDTH_SCALE
        
        # [保留] 用來觀察量化誤差的浮點數，保持小數點輸出
        export_tensor_to_txt(deltaA, "A_answer_exp_int_compare_float.txt", is_int=False)
        # ======================================================================
        delta_B_u = einsum(deltaB_u_Quantization,u, 'b l d_in n, b l d_in -> b l d_in n')
        # 執行選擇性掃描 (Perform selective scan)
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + delta_B_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1) 
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
