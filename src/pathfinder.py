import torch
from torch import nn
from typing import *
from diffusers.models.lora import ConvWithMapper
from diffusers.models.resnet import ResnetBlock2D, Upsample2D
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D
from diffusers.models.vae import Decoder
import numpy as np
from numpy.random import Generator
import utils
import math
import pprint


class Policy():
    def __init__(
        self,
        granularity: Literal[
            'kernel',
            'filter',
        ] = 'kernel',
        layer_selection: Literal[
            'all',
            'layer_begin',  # the conv name that the layer selection starts from (this out-channel included)
            'layer_end',  # the conv name that the layer selection ends at (this in-channel included)
            'layer_range',  # [begin.out, ... , end.in]
        ] = 'all',
        layer_begin: str = 'up_blocks.2.resnets.0.conv1',
        layer_end: str = '',
        use_lora: bool = False,
        lora_rank: int = 4,
        channel_selection: Literal[
            'random',
            'straight',  # straightforward group_num*group_size beginning from start_group*group_size
            'all',  # mainly for filter
        ] = 'all',
        include_bias: bool = False,
        # group
        enable_group: bool = True,  # TODO when False
        continuous_groups: bool = True,  # TODO when False
        chain: bool = True, # for random/critical/max/min
        total_group_num: int = 32,
        group_num: int = 32,
        # for random
        seed: int = 0,
        # for straight
        start_group: int = 11,
        # for random/straight
        conv_in_null_in = True,  # though null in, the crosshair is actually a Union of the horizontal & vertical line
        conv_out_full_out = False,  # actually meaning modify all conv_out
        absolute_perturb: bool = False,
        **ignored_kwargs,
    ):
        self.granularity = granularity
        self.layer_selection = layer_selection
        self.layer_begin = layer_begin
        self.layer_end = layer_end
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.channel_selection = channel_selection
        self.include_bias = include_bias
        self.enable_group = enable_group
        self.continuous_groups = continuous_groups
        self.chain = chain
        self.total_group_num = total_group_num
        self.group_num = group_num
        self.seed = seed
        self.start_group = start_group
        self.conv_in_null_in = conv_in_null_in
        self.conv_out_full_out = conv_out_full_out
        self.absolute_perturb = absolute_perturb

TwoRange = Tuple[int, int]
Indices = List[int]
class Selection():
    in_: TwoRange  # TODO support Indices
    out_: TwoRange
    def __init__(self, ib: int, ie: int, ob: int, oe: int) -> None:
        self.in_ = (ib, ie)
        self.out_ = (ob, oe)
    def __str__(self) -> str:
        return f'[{self.in_}->{self.out_}]'
    def __repr__(self) -> str:
        return self.__str__()
# the type for path
LayerSelections = List[Tuple[str, Selection]]

class Pathfinder():
    # heavily depend on the correct forwarding order of Module.named_children()
    policy: Policy
    rng: Generator
    basename: str
    last_section: Selection
    path: LayerSelections

    def __init__(self, policy: Policy, debug=False, basename='decoder') -> None:
        self.policy = policy
        # random generator
        self.rng = np.random.default_rng(seed=policy.seed)
        self.debug = debug
        self.basename = basename
        self.last_section = None
        self.path = None
    
    def _rm_basename(self, s: str):
        return s.replace(self.basename+'.', '', 1)

    def enter(self, cur_name: str, cur: nn.Module) -> LayerSelections:
        if self.debug:
            print(f'> current: {cur_name}')
        ret: LayerSelections = []
        if isinstance(cur, ConvWithMapper):
            out_size, in_size = cur.weight.data.shape[:2]
            if self.policy.channel_selection == 'all':
                sel = Selection(0, in_size, 0, out_size)
            # straight
            elif self.policy.channel_selection == 'straight':
                def straight(channel_size: int):
                    group_size = channel_size // self.policy.total_group_num
                    begin_index = self.policy.start_group * group_size
                    end_index = begin_index + self.policy.group_num * group_size
                    return begin_index, end_index
                if self.policy.conv_in_null_in and 'conv_in' in cur_name:
                    ib, ie = 0, 0
                    ob, oe = straight(out_size)
                elif self.policy.conv_out_full_out and 'conv_out' in cur_name:
                    ib, ie = straight(in_size)
                    ob, oe = 0, out_size
                else:
                    ib, ie = straight(in_size)
                    ob, oe = straight(out_size)
                sel = Selection(ib, ie, ob, oe)
            # random
            elif self.policy.channel_selection == 'random':
                assert self.policy.enable_group and self.policy.continuous_groups
                def random(channel_size: int):
                    group_size = channel_size // self.policy.total_group_num
                    start_group = self.rng.integers(
                        low=0, 
                        high=self.policy.total_group_num-self.policy.group_num+1,
                        size=1,
                    ).item()
                    begin_index = start_group * group_size
                    end_index = begin_index + self.policy.group_num * group_size
                    return begin_index, end_index
                if self.policy.conv_in_null_in and 'conv_in' in cur_name:
                    ib, ie = 0, 0
                    ob, oe = random(out_size)
                elif self.policy.conv_out_full_out and 'conv_out' in cur_name:
                    ib, ie = self.last_section.out_ if self.policy.chain \
                        else random(in_size)
                    ob, oe = 0, out_size
                else:
                    ib, ie = self.last_section.out_ if self.policy.chain \
                        else random(in_size)
                    ob, oe = random(out_size)
                sel = Selection(ib, ie, ob, oe)
            else: raise Exception()
            self.last_section = sel
            if self.debug:
                print(f'  > New sel: {sel}')  # TODO debug
            return [(cur_name, sel)]
        elif isinstance(cur, Upsample2D):
            pass  # real pass
        elif isinstance(cur, ResnetBlock2D):
            if self.policy.chain and self.policy.channel_selection in ['random']:
                # for shortcut
                sib, sie = self.last_section.out_
                ret += self.enter(cur_name+'.conv1', cur.conv1)
                ret += self.enter(cur_name+'.conv2', cur.conv2)
                sob, soe = self.last_section.out_
                if utils.attr(cur, 'conv_shortcut'):
                    sel = Selection(sib, sie, sob, soe)
                    if self.debug:
                        print(f'  > New sel for shortcut: {sel}')  # TODO debug
                    temp = [(cur_name+'.conv_shortcut', sel)]
                    ret += temp
                return ret
            else: 
                pass  # real pass
        elif isinstance(cur, UNetMidBlock2D):
            pass  # real pass
        elif isinstance(cur, UpDecoderBlock2D):
            pass  # real pass
        elif isinstance(cur, Decoder):
            pass  # real pass
        else:
            if self.debug:
                print(f'  > Passed {cur_name}, type: {type(cur)}')
        # get children
        nc = [x for x in cur.named_children()]
        # make sure right order, mid before up, TODO: test
        if isinstance(cur, Decoder):
            mid_part = [i for i in nc if 'mid_block' in i[0]]
            not_mid_part = [i for i in nc if 'mid_block' not in i[0]]
            nc = [not_mid_part[0]] + mid_part + not_mid_part[1:] if not_mid_part else mid_part
        # recursive
        for suffix, next in nc:
            next_name = f'{cur_name}.{suffix}'
            ret += self.enter(next_name, next)
        # do layer selection
        if isinstance(cur, Decoder):
            if self.policy.layer_selection=='layer_begin':
                for i in range(len(ret)):
                    if self._rm_basename(ret[i][0])!=self.policy.layer_begin:
                        ret[i][1].in_ = (0, 0)
                        ret[i][1].out_ = (0, 0)
                    else:
                        ret[i][1].in_ = (0, 0)
                        break
            elif self.policy.layer_selection=='layer_end':
                end_index = -1
                for i in range(len(ret)):
                    if self._rm_basename(ret[i][0])==self.policy.layer_end:
                        ret[i][1].out_ = (0, 0)
                        end_index = i
                        break
                for i in range(end_index+1, len(ret)):
                    ret[i][1].in_ = (0, 0)
                    ret[i][1].out_ = (0, 0)
            elif self.policy.layer_selection=='layer_range':
                begin_index = -1
                for i in range(len(ret)):
                    if self._rm_basename(ret[i][0])!=self.policy.layer_begin:
                        ret[i][1].in_ = (0, 0)
                        ret[i][1].out_ = (0, 0)
                    else:
                        ret[i][1].in_ = (0, 0)
                        begin_index = i
                        break
                end_index = -1
                for i in range(begin_index+1, len(ret)):
                    if self._rm_basename(ret[i][0])==self.policy.layer_end:
                        ret[i][1].out_ = (0, 0)
                        end_index = i
                        break
                for i in range(end_index+1, len(ret)):
                    ret[i][1].in_ = (0, 0)
                    ret[i][1].out_ = (0, 0)
        return ret
    
    def explore(self, model: nn.Module)->None:
        self.path = self.enter(self.basename, model)

    def print_path(self):
        # talk about policy as well
        pprint.pprint(self.policy.__dict__, width=1)
        for name, sel in self.path:
            print(f'{name}: {sel}')

    # return total_size
    def init_model(self, model: nn.Module) -> int:
        # get path_dict
        removed_model_name = [(self._rm_basename(s), sel) for s, sel in self.path]
        path_dict = {k: v for k, v in removed_model_name}
        # for each
        convs: List[Tuple[str, ConvWithMapper]] = utils.get_convs(model)
        # the partition beginning
        cur = 0
        for name, conv in convs:
            sel = path_dict.get(name)
            if sel is None:
                raise Exception(f'no selection for {name} in path dict')

            if self.policy.granularity == 'filter' or self.policy.use_lora:
                # only care about sel.out_
                sel.in_ = (0, 0)
            conv.absolute_perturb = self.policy.absolute_perturb

            conv.in_msg_start = sel.in_[0]
            conv.in_msg_end = sel.in_[1]
            conv.out_msg_start = sel.out_[0]
            conv.out_msg_end = sel.out_[1]

            conv.in_msg_size = conv.in_msg_end - conv.in_msg_start
            conv.out_msg_size = conv.out_msg_end - conv.out_msg_start
            conv.in_weight_shape = (conv.out_channels, conv.in_msg_size, 1, 1)
            if self.policy.granularity == 'kernel' or self.policy.use_lora:
                conv.out_weight_shape = (conv.out_msg_size, conv.in_channels, 1, 1)
            elif self.policy.granularity == 'filter':
                conv.out_weight_shape = (conv.out_msg_size, 1, 1, 1)
            mod_weight_size = (math.prod(conv.out_weight_shape) +
                            math.prod(conv.in_weight_shape) -
                            conv.out_msg_size * conv.in_msg_size)
            
            if self.policy.include_bias:
                conv.mod_bias_shape = (conv.out_channels,)
                mod_bias_size = math.prod(conv.mod_bias_shape)
            else:
                conv.mod_bias_shape = None
                mod_bias_size = 0
            
            conv.mod_weight_size = mod_weight_size
            conv.mod_bias_size = mod_bias_size

            conv.out_weight_size = math.prod(conv.out_weight_shape)
            conv.in_weight_shapeU = torch.Size((conv.out_msg_start, conv.in_msg_size, 1, 1))
            conv.in_weight_shapeD = torch.Size((conv.out_channels - conv.out_msg_end, conv.in_msg_size, 1, 1))
            conv.in_weight_sizeU = math.prod(conv.in_weight_shapeU)
            conv.in_weight_sizeD = math.prod(conv.in_weight_shapeD)

            if self.policy.use_lora:
                conv.lora_a_shape = (conv.out_msg_size, self.policy.lora_rank, 1, 1)
                conv.lora_b_shape = (self.policy.lora_rank, conv.in_channels, 1, 1)
                conv.lora_a_size = math.prod(conv.lora_a_shape)
                conv.lora_b_size = math.prod(conv.lora_b_shape)
                partition_size = conv.lora_a_size + conv.lora_b_size
                if partition_size != 0:
                    conv.use_lora = True
            else:
                partition_size = mod_weight_size + mod_bias_size
            conv.partition = (cur, cur + partition_size)
            cur += partition_size
        return cur