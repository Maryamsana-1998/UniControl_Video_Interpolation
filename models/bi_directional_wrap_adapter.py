import torch
import torch as th
import torch.nn as nn
from models.softsplat import softsplat
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
)
from adapter_utils import *

class Bi_Dir_FeatureExtractor(nn.Module):

    def __init__(self, inject_channels, dims=2):
        super().__init__()
        self.first_pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.last_pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.wrapper = FeatureWarperSoftsplat()
        self.extractors = nn.ModuleList([
            LocalTimestepEmbedSequential(
                conv_nd(dims, 128, inject_channels[0], 3, padding=1, stride=2),
                nn.SiLU()
            ),
            LocalTimestepEmbedSequential(
                conv_nd(dims, inject_channels[0], inject_channels[1], 3, padding=1, stride=2),
                nn.SiLU()
            ),
            LocalTimestepEmbedSequential(
                conv_nd(dims, inject_channels[1], inject_channels[2], 3, padding=1, stride=2),
                nn.SiLU()
            ),
            LocalTimestepEmbedSequential(
                conv_nd(dims, inject_channels[2], inject_channels[3], 3, padding=1, stride=2),
                nn.SiLU()
            )
        ])
        self.zero_convs = nn.ModuleList([
            zero_module(conv_nd(dims, inject_channels[0], inject_channels[0], 3, padding=1)),
            zero_module(conv_nd(dims, inject_channels[1], inject_channels[1], 3, padding=1)),
            zero_module(conv_nd(dims, inject_channels[2], inject_channels[2], 3, padding=1)),
            zero_module(conv_nd(dims, inject_channels[3], inject_channels[3], 3, padding=1))
        ])
    
    def forward(self, local_conditions, flow):

        first_frame = local_conditions[:,3:]
        last_frame = local_conditions[:,:3]
        flow_fwd = flow[:,:2]
        flow_bwd = flow[:,2:]

        print('print shapes:',first_frame.shape, last_frame.shape, flow_fwd.shape,flow_bwd.shape)

        first_features = self.first_pre_extractor(first_frame,None)
        last_features = self.last_pre_extractor(last_frame,None)

        wrapped_first = self.wrapper(first_features, flow_fwd)
        wrapped_last = self.wrapper(last_features,flow_bwd)

        local_features = torch.cat([wrapped_first,wrapped_last],dim=0)
        
        assert len(self.extractors) == len(self.zero_convs)
        output_features = []
        for idx in range(len(self.extractors)):
            local_features = self.extractors[idx](local_features, None)
            output_feature = self.zero_convs[idx](local_features)
            output_features.append(output_feature)
        return output_features