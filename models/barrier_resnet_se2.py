import copy
import math
import warnings
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Callable, List
from torch import Tensor
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.transformer import _get_activation_fn
from torch.nn.parameter import Parameter
from torch.nn.functional import _mha_shape_check, _in_projection_packed, _in_projection
from torch.nn import Module, Dropout, Linear, LayerNorm
import torch.nn.functional as F

# ==========================================
# Custom Attention Implementation
# ==========================================
def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """
    Custom scaled dot-product attention to extract attention weights.
    
    Args:
        q: [batch*num_heads, tgt_len, head_dim]
        k: [batch*num_heads, src_len, head_dim]
        v: [batch*num_heads, src_len, head_dim]
        attn_mask: optional mask
        dropout_p: dropout probability
    
    Returns:
        attn_output: [batch*num_heads, tgt_len, head_dim]
        attn_weights: [batch*num_heads, tgt_len, src_len]
    """
    head_dim = q.size(-1)
    
    # Calculate attention scores
    attn_scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask
    
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Preserve weights before dropout for visualization/analysis
    attn_weights_for_return = attn_weights.detach().clone()
    
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    attn_output = torch.bmm(attn_weights, v)
    
    return attn_output, attn_weights_for_return

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops, query, key, value, embed_dim_to_check, num_heads,
            in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn,
            dropout_p, out_proj_weight, out_proj_bias, training=training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v,
            average_attn_weights=average_attn_weights,
        )
    
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)
    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
    
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim
    
    if use_separate_proj_weight:
        assert key.shape[:2] == value.shape[:2]
    else:
        assert key.shape == value.shape
    
    if not use_separate_proj_weight:
        assert in_proj_weight is not None
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None
        assert k_proj_weight is not None
        assert v_proj_weight is not None
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            if attn_mask.shape != (bsz * num_heads, tgt_len, src_len):
                raise RuntimeError(f"Invalid 3D attn_mask shape: {attn_mask.shape}")
    
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)
    
    if bias_k is not None and bias_v is not None:
        assert static_k is None
        assert static_v is None
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None
    
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        v = static_v
    
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    
    src_len = k.size(1)
    
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len)
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask.to(query.dtype).masked_fill(key_padding_mask, float('-inf')) if key_padding_mask.dtype == torch.bool else key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
    
    if not training:
        dropout_p = 0.0
    
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        if not is_batched:
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            attn_output = attn_output.squeeze(1)
        return attn_output, None

class MultiheadAttention(Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        
        is_batched = query.dim() == 3
        
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, 
                need_weights=need_weights, 
                attn_mask=attn_mask, 
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, 
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, 
                average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, 
                need_weights=need_weights, 
                attn_mask=attn_mask, 
                average_attn_weights=average_attn_weights)
        
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

# ==========================================
# Modified Transformer Components
# ==========================================
class CustomTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        self.return_attention = False

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None, **kwargs) -> Tensor:

        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask

        if not (torch.jit.is_scripting() or torch.jit.is_tracing()):
            if src.dim() != 3:
                raise RuntimeError(f"{self.__class__.__name__}.forward() requires a 3D tensor input.")

        all_attentions = [] if self.return_attention else None

        for mod in self.layers:
            if self.return_attention:
                output, attn_weights = mod(
                    output, 
                    src_mask=mask, 
                    src_key_padding_mask=src_key_padding_mask_for_layers,
                    is_causal=is_causal, 
                    need_weights=True,
                    average_attn_weights=False,
                    **kwargs
                )
                all_attentions.append(attn_weights)
            else:
                output = mod(
                    output, 
                    src_mask=mask, 
                    src_key_padding_mask=src_key_padding_mask_for_layers,
                    is_causal=is_causal, 
                    **kwargs
                )

        if self.norm is not None:
            output = self.norm(output)

        if self.return_attention:
            return output, all_attentions
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None, need_weights: bool = False,
                average_attn_weights: bool = True, **kwargs):
        
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights
        )
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if need_weights:
            return src, attn_weights
        return src

# ==========================================
# Main Architecture: CT-DiffNet
# ==========================================
class CNNTransformer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 64,
        num_cnn_blocks: int = 2,
        patch_size: int = 2,
        emb_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 256,
    ):
        super().__init__()
        
        # 1. CNN Backbone
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
        )
        
        cnn_layers = []
        for _ in range(num_cnn_blocks):
            cnn_layers += [
                nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm3d(hidden_channels),
                nn.GELU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
            ]
        self.cnn = nn.Sequential(*cnn_layers)
       
        # 2. Patchify & Linear Projection
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.proj = nn.Linear(hidden_channels * patch_size**3, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, emb_dim))
       
        # 3. Transformer Encoder
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu"
        )
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
       
        # 4. Regression Head
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        )
        
        self.return_attention = False
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (B, in_channels, D, D, D)
        
        Returns:
            Output regression value (B,)
            Optional: List of attention weights if self.return_attention is True.
        """
        B = x.size(0)
        
        # CNN feature extraction
        x = self.stem(x)
        x = self.cnn(x)
        
        # Patchify
        _, C, Dp, _, _ = x.shape
        P = Dp // self.patch_size
        x = x.view(B, C, P, self.patch_size, P, self.patch_size, P, self.patch_size)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        num_patches = P**3
        x = x.view(B, num_patches, C * self.patch_size**3)
       
        # Linear projection & Positional encoding
        x = self.proj(x)
        
        if self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.emb_dim, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        x = x + self.pos_embed
       
        # Transformer processing
        x = x.permute(1, 0, 2) 
        
        self.transformer.return_attention = self.return_attention
        
        if self.return_attention:
            x, all_attentions = self.transformer(x)
        else:
            x = self.transformer(x)
            all_attentions = None
        
        x = x.permute(1, 0, 2)
       
        # Global average pooling & Regression
        x = x.mean(dim=1)
        output = self.head(x).squeeze(-1)
        
        if self.return_attention:
            return output, all_attentions
        return output