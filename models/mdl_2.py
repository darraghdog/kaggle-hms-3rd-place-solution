#Several SqueezeFormer components where copied/ adapted from https://github.com/upskyy/Squeezeformer/

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
import typing
from torch import Tensor
import math
import numpy as np
import timm
import json
from transformers.models.speech_to_text import Speech2TextConfig, Speech2TextForConditionalGeneration
from transformers.models.speech_to_text.modeling_speech_to_text import shift_tokens_right, Speech2TextDecoder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding
from timm.layers.norm_act import BatchNormAct2d
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from torch.distributions import Beta



def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")

class Decoder(nn.Module):
    def __init__(self, decoder_config):
        super(Decoder, self).__init__()
        
        self.config = decoder_config
        self.decoder = Speech2TextDecoder(decoder_config) 
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        
        self.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.decoder_end_token_id= decoder_config.eos_token_id
        
    def forward(self,x, labels=None, attention_mask = None, encoder_attention_mask = None):
        
        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
            
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=x, 
                                       attention_mask = attention_mask,
                                       encoder_attention_mask = encoder_attention_mask)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        return lm_logits
            
    def generate(self, x, max_new_tokens=33, encoder_attention_mask=None):

        decoder_input_ids = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.long).fill_(self.decoder_start_token_id)
        for i in range(max_new_tokens-1):  
            decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=x, encoder_attention_mask=encoder_attention_mask)
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            decoder_input_ids = torch.cat([decoder_input_ids,logits.argmax(2)[:,-1:]],dim=1)

            if torch.all((decoder_input_ids==self.decoder_end_token_id).sum(-1) > 0):
                break
                
        return decoder_input_ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        
        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        
        return x


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class DepthwiseConv2d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 2-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    # mask_pad = mask.bool().unsqueeze(1)
    def forward(self, x, mask_pad):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        Reference for masking : https://github.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch/wenet/transformer/convolution.py#L26
        """
        # mask batch padding
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        # torch.Size([4, 128, 384])
        x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
        mask_bn = mask_pad.view(-1)
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        '''
        x = self.bn(x)
        '''
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x



def make_scale(encoder_dim):
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None,None,:])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None,None,:])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerBlock, self).__init__()
        
        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)
        
        '''
        self.mhsa = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,)
        encoder_dim = 144
        num_attention_heads = 4
        attention_dropout_p = 0.1
        self.mhsa_whisper = WhisperAttention(embed_dim = encoder_dim,\
                                       num_heads = num_attention_heads,\
                                       dropout = attention_dropout_p,\
                                       is_decoder = False,\
                                       bias = True)
        '''       
        
        
        self.mhsa_llama = LlamaAttention(LlamaConfig(hidden_size = encoder_dim, 
                                       num_attention_heads = num_attention_heads, 
                                       max_position_embeddings = 384))
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        
        self.ff_mhsa = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        
        
        # Attention_mask = (bsz, self.num_heads, q_len, kv_seq_len)   
            
            
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)
        
        '''
        self.mhsa = self.encoder.blocks[0].mhsa
        self.ln_mhsa = self.encoder.blocks[0].ln_mhsa
        self.ln_ff_mhsa = self.encoder.blocks[0].ln_ff_mhsa
        self.ln_conv = self.encoder.blocks[0].ln_conv
        self.ln_ff_conv = self.encoder.blocks[0].ln_ff_conv
        self.ff_mhsa = self.encoder.blocks[0].ff_mhsa
        self.ln_mhsa = self.encoder.blocks[0].ln_mhsa
        self.conv = self.encoder.blocks[0].conv
        self.ff_conv = self.encoder.blocks[0].ff_conv
        '''

    def forward(self, x, cos, sin, mask):
        mask_pad = ( mask).long().bool().unsqueeze(1)
        mask_pad = ~( mask_pad.permute(0, 2,1) * mask_pad)
        mask_flat = mask.reshape(-1).bool()
        bs, slen, nfeats = x.shape
        
        residual = x
        x = x * self.scale_mhsa.to(x.dtype) + self.bias_mhsa.to(x.dtype)
        x = residual + self.mhsa_llama(x, cos, sin, attention_mask = mask_pad.unsqueeze(1) )[0]
        # Skip pad #1
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x) #casts to fp32
        residual = x
        #32
        x = x * self.scale_ff_mhsa.to(x.dtype) + self.bias_ff_mhsa.to(x.dtype)
        #32
        x = residual + self.ff_mhsa(x) 
        #32
        x = self.ln_ff_mhsa(x) #casts to fp32
        
        
        # Unskip pad #1
#         print(x_skip[mask_flat].dtype, x[0].dtype)
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.reshape(bs, slen, nfeats)
        residual = x
        # torch.Size([16, 384, 128])
        x = x * self.scale_conv.to(x.dtype) + self.bias_conv.to(x.dtype)
        x = residual + self.conv(x, mask_pad = mask.bool().unsqueeze(1))
        # Skip pad #2
        x_skip = x.reshape(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        
        x = self.ln_conv(x)
        
        
        residual = x
        x = x * self.scale_ff_conv.to(x.dtype) + self.bias_ff_conv.to(x.dtype)
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)
        
        # Unskip pad #2
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)  
        
        
        return x
    
class TimeReductionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super(TimeReductionLayer, self).__init__()
        self.sequential = nn.Sequential(
            DepthwiseConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            Swish(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        output_lengths = input_lengths >> 1
        output_lengths -= 1
        return outputs, output_lengths
    
class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_indices: list = [2,4],
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None
        self.time_reduction_layer = TimeReductionLayer()
        self.time_reduction_proj = nn.Linear((encoder_dim - 1) // 2, encoder_dim)
        
        self.reduce_layer_indices = reduce_layer_indices

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )
          
            '''
            cos, sin = self.cos, self.sin
            '''
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        outputs = x
        output_lengths = mask.sum(-1)
        
        for idx, layer in enumerate(self.blocks):
            
            if idx in self.reduce_layer_indices:
                outputs, output_lengths = self.time_reduction_layer(outputs, output_lengths)
                outputs = self.time_reduction_proj(outputs)
            # Rebuild mask
            seq = torch.arange(outputs.shape[1], device=x.device)
            outputs_mask = (seq < output_lengths.unsqueeze(-1)).long()
            
            outputs = layer(outputs, cos, sin, outputs_mask)

        return outputs, outputs_mask





class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        '''
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        '''
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:, :, :kv_seq_len], sin[:, :, :kv_seq_len])
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights.masked_fill_(attention_mask, torch.finfo(attn_weights.dtype).min)


        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.squeeze(1)
    k_embed = k_embed.squeeze(1)
    return q_embed, k_embed



def get_lm_type(lm):
    if 'left_hand' in lm:
        t = 1
    elif 'right_hand' in lm:
        t = 2    
    elif 'face' in lm:
        t = 3  
    elif 'pose' in lm:
        t = 4  
    return t




class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class cbam_block(nn.Module):

    def __init__(self, channel, ratio=4, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

def channel_shuffle(x, groups):

    # input shape: [batch_size, channels, H, W]
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

class ChannelShuffle(nn.Module):

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("The number of channels must be divisible by the number of groups.")
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)
    
    
'''
        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)
'''

class CNN(nn.Module):

    def __init__(self, F1: int, classes_num: int, D: int = 2, final_dropout=0.25):

        super(CNN, self).__init__()
        self.drop_out = final_dropout
        self.att = cbam_block(D * F1)
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 16),
                stride=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8))
        )
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((7, 7, 0, 0)),
            nn.Conv2d(
                in_channels=F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels= D * F1,
                out_channels=D * F1,
                kernel_size=(1, 16),
                stride=(1, 2),
                bias=False,
                groups=D * F1,
            ),
            nn.BatchNorm2d(D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(D * F1),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * D * F1, # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(3, 1),
                stride=(1, 1),
                groups=D * D * F1,
                bias=False
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1, # D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(D * F1), # D * D * F1
            # ChannelShuffle(D * D * F1, 4), # D * D * F1
        )
        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels= D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            Swish(),#GLU(dim=1),)
            nn.Conv2d(
                in_channels= D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 8),
                stride=(1, 1),
                bias=False,
                groups=D * D * F1,
            ),
            nn.BatchNorm2d(D * D * F1),
            Swish(),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
                groups=4,
            ),
            nn.BatchNorm2d(D * F1),
            nn.AvgPool2d((1, 2)),
            # nn.ReLU(inplace=True),
            # ChannelShuffle(D * D * F1, 4), # D * D * F1
        )
        '''
        self.block_4 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * F1,
                kernel_size=(1, 8),
                stride=(1, 1),
                groups=D * D * F1,
                bias=False
            ),
            nn.BatchNorm2d(D * D * F1),
            nn.Conv2d(
                in_channels=D * D * F1,
                out_channels=D * D * D * F1,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=4,
                bias=False
            ),
            nn.BatchNorm2d(D * D * D * F1),
            nn.ReLU(inplace=True),
#             nn.AvgPool2d((1, 16))
        )
        '''

    def forward(self, x):

        # print(x.shape)
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
#         x1 = x.mean(2, keepdim=True)
#         x2 = torch.norm(x, p=2, dim=2, keepdim=True)
#         x3 = torch.norm(x, p=np.inf, dim=2, keepdim=True)
#         x = torch.cat([x1, x2, x3], 2)
        
        # print(x.shape)
        x = self.att(x)
        
        # print(x.shape)
        x = self.block_3(x)
        
        # print(x.shape)
        x = self.block_4(x)

        return x



'''
self = Net(cfg)
self = self.cnn

self.cnn(x).shape
'''

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.cfg = cfg
        self.n_classes = len(self.cfg.targets)
        
        D = 2
        F1 = cfg.encoder_config.input_dim // 2
        self.cnn = CNN(F1 = F1, classes_num = self.n_classes, D = D, final_dropout=0.25)
        encoder_dim = D * D * F1
        
        rotary_emb = LlamaRotaryEmbedding(cfg.encoder_config.encoder_dim//cfg.encoder_config.num_attention_heads, max_position_embeddings=cfg.max_len)
        self.cos = torch.nn.parameter.Parameter(rotary_emb.cos_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        self.sin = torch.nn.parameter.Parameter(rotary_emb.sin_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)

        while len(self.cos.shape)<4: self.cos = torch.nn.parameter.Parameter(self.cos.unsqueeze(0))
        while len(self.sin.shape)<4: self.sin = torch.nn.parameter.Parameter(self.sin.unsqueeze(0))        
        
        self.encoder = SqueezeformerEncoder(
                      input_dim=cfg.encoder_config.input_dim,
                      reduce_layer_indices = cfg.encoder_config.reduce_layer_indices,
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      input_dropout_p= cfg.encoder_config.input_dropout_p,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
        
        
        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(cfg.encoder_config.encoder_dim, self.n_classes)
        self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self.loss_fn2 = torch.nn.KLDivLoss(reduction='none')
        print('n_params:',count_parameters(self))

    def forward(self, batch, debug = False):

        x = batch['input'].float().unsqueeze(-1) # bs, seq_len, n_landmarks, 1
        mask = torch.ones_like(x[:,:,0,0], dtype = torch.long)
        y = batch["target"].float()
        
        
        if (self.training):
            for tt in range(x.shape[0]):
                if self.cfg.aug_drop_spec_prob > np.random.random():
                    drop_ct = np.random.randint(1, 1+self.cfg.aug_drop_spec_max)
                    drop_idx = np.random.choice(np.arange(x.shape[2]), drop_ct)
                    x[tt, :, drop_idx] = 0.
        x = x.permute(0,3,2,1)
        
        x = self.cnn(x)
        
        
        x = x.permute(0,1,3,2)
        bs, slen, hf, wf = x.shape
        x = x.reshape(bs, slen * hf, wf)
        x = x.unsqueeze(-1)
        
        x = self.global_pool(x)[:,:,0,0]
        x = x.reshape(bs, slen, hf)
        x = x.permute(0, 2, 1)
        
        # Input torch.Size([8, 625, 128])
        x, mask = self.encoder(x, self.cos, self.sin, mask)
        
        x = x.unsqueeze(1).permute(0, 3, 1, 2)
        x = self.global_pool(x)
        x = x[:,:,0,0]

        logits = self.head(x)
        if 'label_weight' in batch:
            wts = batch['label_weight'].float()
            wtsunq = torch.unique(wts)
            loss = sum([wt * self.loss_fn(F.log_softmax(logits[wts==wt], dim=1),y[wts==wt]) for wt in wtsunq])

        else:
            loss = self.loss_fn(F.log_softmax(logits, dim=1),y)
        outputs = {}
        outputs['loss'] = loss
        if not self.training:
            outputs["logits"] = logits
 

        return outputs
