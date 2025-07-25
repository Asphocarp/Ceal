# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTANT:                                                      #
###################################################################
# ----------------------------------------------------------------#
# This file is deprecated and will be removed soon                #
# (as soon as PEFT will become a required dependency for LoRA)    #
# ----------------------------------------------------------------#
###################################################################

from typing import *

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import logging
from ..utils.import_utils import is_transformers_available


if is_transformers_available():
    from transformers import CLIPTextModel, CLIPTextModelWithProjection


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            mlp_mod = layer.mlp
            name = f"text_model.encoder.layers.{i}.mlp"
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(f"do not know how to get mlp modules for: {text_encoder.__class__.__name__}")

    return mlp_modules


def adjust_lora_scale_text_encoder(text_encoder, lora_scale: float = 1.0):
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        if isinstance(attn_module.q_proj, PatchedLoraProjection):
            attn_module.q_proj.lora_scale = lora_scale
            attn_module.k_proj.lora_scale = lora_scale
            attn_module.v_proj.lora_scale = lora_scale
            attn_module.out_proj.lora_scale = lora_scale

    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        if isinstance(mlp_module.fc1, PatchedLoraProjection):
            mlp_module.fc1.lora_scale = lora_scale
            mlp_module.fc2.lora_scale = lora_scale


class PatchedLoraProjection(torch.nn.Module):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        super().__init__()
        from ..models.lora import LoRALinearLayer

        self.regular_linear_layer = regular_linear_layer

        device = self.regular_linear_layer.weight.device

        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        self.lora_linear_layer = LoRALinearLayer(
            self.regular_linear_layer.in_features,
            self.regular_linear_layer.out_features,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype,
            rank=rank,
        )

        self.lora_scale = lora_scale

    # overwrite PyTorch's `state_dict` to be sure that only the 'regular_linear_layer' weights are saved
    # when saving the whole text encoder model and when LoRA is unloaded or fused
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if self.lora_linear_layer is None:
            return self.regular_linear_layer.state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

    def _fuse_lora(self, lora_scale=1.0, safe_fusing=False):
        if self.lora_linear_layer is None:
            return

        dtype, device = self.regular_linear_layer.weight.data.dtype, self.regular_linear_layer.weight.data.device

        w_orig = self.regular_linear_layer.weight.data.float()
        w_up = self.lora_linear_layer.up.weight.data.float()
        w_down = self.lora_linear_layer.down.weight.data.float()

        if self.lora_linear_layer.network_alpha is not None:
            w_up = w_up * self.lora_linear_layer.network_alpha / self.lora_linear_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.regular_linear_layer.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_linear_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self.lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.regular_linear_layer.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self.lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.regular_linear_layer.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, input):
        if self.lora_scale is None:
            self.lora_scale = 1.0
        if self.lora_linear_layer is None:
            return self.regular_linear_layer(input)
        return self.regular_linear_layer(input) + (self.lora_scale * self.lora_linear_layer(input))


class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRAConv2dLayer(nn.Module):
    r"""
    A convolutional layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
    ):
        super().__init__()

        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((w_orig.shape))
        fused_weight = w_orig + (lora_scale * fusion)

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        maps: torch.Tensor = None,
        scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            original_outputs = F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return original_outputs + (scale * self.lora_layer(hidden_states))


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out


class ConvWithMapper(nn.Conv2d):
    '''
    It functions like normal conv before init_mapper.
    '''
    # all to be initiated by pathfinder
    in_msg_start: int
    in_msg_end: int
    out_msg_start: int
    out_msg_end: int

    in_msg_size: int
    out_msg_size: int

    out_weight_shape: Tuple[int]
    in_weight_shape: Tuple[int]
    # None for no bias modulation
    mod_bias_shape: Tuple[int] | None

    out_weight_size: int
    in_weight_shapeU: torch.Size
    in_weight_shapeD: torch.Size
    in_weight_sizeU: int
    in_weight_sizeD: int

    mod_weight_size: int
    mod_bias_size: int

    # the map partition of this conv
    partition: Tuple[int, int]

    # when using lora, use out_weight_map (out, in, 1, 1) only
    # out means out_msg_size, typically equal to out_channel_num
    use_lora: bool = False
    lora_a_shape: Tuple[int]  # (out, r, 1, 1)
    lora_b_shape: Tuple[int]  # (r,  in, 1, 1)
    lora_a_size: int
    lora_b_size: int

    test_collusion: bool = False
    absolute_perturb: bool = False

    def __init__(self,
                 *args,
                 the_force:float=0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.the_force = the_force

    def spawn(self, maps):
        # reshape
        in_weight_mapD, in_weight_mapU, out_weight_map, bias_map \
            = self.reshape_map(maps)

        # modulate (affine)
        woi, wui, wdi = \
            out_weight_map[0], in_weight_mapU[0], in_weight_mapD[0]
        modulated_weight = self.weight.clone()
        modulated_weight[self.out_msg_start:self.out_msg_end] *= 1 + self.the_force * woi
        modulated_weight[:self.out_msg_start, self.in_msg_start:self.in_msg_end] *= 1 + self.the_force * wui
        modulated_weight[self.out_msg_end:, self.in_msg_start:self.in_msg_end] *= 1 + self.the_force * wdi
        # if bias_map is not None:
        #     bi = bias_map[0]
        #     modulated_bias = self.bias + self.the_force * bi
        # else:
        #     modulated_bias = self.bias
        bi = bias_map[0]
        modulated_bias = self.bias + self.the_force * bi
        self.weight = nn.Parameter(modulated_weight)
        self.bias = nn.Parameter(modulated_bias)

    def reshape_map(self, maps: Tensor)->Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        using view, so data is not copied
        """
        zeros = [0 for _ in range(maps.shape[0])]

        if self.test_collusion:
            # assert maps.shape[0] > 1  # at least 2 attackers
            zeros = [0]

        if self.use_lora:
            assert self.partition[1]-self.partition[0] == self.lora_a_size+self.lora_b_size
            if self.lora_a_size == 0 or self.lora_b_size == 0:
                return zeros, zeros, zeros, zeros
            ab_map = maps[:, self.partition[0]:self.partition[1]]
            lora_a = ab_map[:, :self.lora_a_size].view(-1, *self.lora_a_shape[:2])
            lora_b = ab_map[:, self.lora_a_size:].view(-1, *self.lora_b_shape[:2])
            out_weight_map = (lora_a @ lora_b).view(-1, *self.out_weight_shape)
            if self.test_collusion:
                out_weight_map = out_weight_map.mean(0, keepdim=True)
            return zeros, zeros, out_weight_map, zeros
            
        # not using lora:
        assert self.partition[1]-self.partition[0] == self.mod_weight_size+self.mod_bias_size
        weight_map = maps[:, self.partition[0]:self.partition[0]+self.mod_weight_size]
        if self.test_collusion:
            weight_map = weight_map.mean(0, keepdim=True)
        if self.out_weight_size != 0:
            out_weight_map = weight_map[:, :self.out_weight_size].view(-1, *self.out_weight_shape)
        else:
            out_weight_map = zeros
        if self.in_weight_sizeU != 0:
            in_weight_mapU = (weight_map[:, self.out_weight_size:self.out_weight_size + self.in_weight_sizeU]
                              .view(-1, *self.in_weight_shapeU))
        else:
            in_weight_mapU = zeros
        if self.in_weight_sizeD != 0:
            in_weight_mapD = (weight_map[:, self.out_weight_size + self.in_weight_sizeU:]
                              .view(-1, *self.in_weight_shapeD))
        else:
            in_weight_mapD = zeros

        if self.mod_bias_size != 0:
            bias_map = maps[:, self.partition[0]+self.mod_weight_size:self.partition[1]]
            if self.test_collusion:
                bias_map = bias_map.mean(0, keepdim=True)
            bias_map_ret = bias_map.view(-1, *self.mod_bias_shape)
        else:
            bias_map_ret = zeros
        return in_weight_mapD, in_weight_mapU, out_weight_map, bias_map_ret


    def forward(self, x, scale=1.0,
                maps=None,
                ) -> torch.Tensor:
        """
        msg: None => just normal conv forward
             'use_cached_map' => using cached maps in mappers to modulate
             else => message to modulate
        """

        # act as normal when msg is None
        if maps is None:
            return F.conv2d(
                x, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups,
            )

        # reshape
        in_weight_mapD, in_weight_mapU, out_weight_map, bias_map \
            = self.reshape_map(maps)

        # per element in a batch
        out = []
        for xi, bi, woi, wui, wdi in \
                zip(x, bias_map, out_weight_map, in_weight_mapU, in_weight_mapD):
            modulated_weight = self.weight.clone()
            if not self.absolute_perturb:
                modulated_weight[self.out_msg_start:self.out_msg_end] *= 1 + self.the_force * woi
                modulated_weight[:self.out_msg_start, self.in_msg_start:self.in_msg_end] *= 1 + self.the_force * wui
                modulated_weight[self.out_msg_end:, self.in_msg_start:self.in_msg_end] *= 1 + self.the_force * wdi
            else:
                modulated_weight[self.out_msg_start:self.out_msg_end] += self.the_force * woi
                modulated_weight[:self.out_msg_start, self.in_msg_start:self.in_msg_end] += self.the_force * wui
                modulated_weight[self.out_msg_end:, self.in_msg_start:self.in_msg_end] += self.the_force * wdi
            modulated_bias = self.bias + self.the_force * bi
            yi = F.conv2d(
                xi.unsqueeze(0),
                modulated_weight, modulated_bias,
                self.stride, self.padding, self.dilation, self.groups
                ).squeeze()
            out += [yi]
        out = torch.stack(out, 0)

        return out


# LoRACompatibleConvWithMapper = ConvWithMapper
# TODO lora compatible conv with mapper, for not USE_PEFT_BACKEND