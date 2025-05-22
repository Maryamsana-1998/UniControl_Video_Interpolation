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
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock


class LocalTimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, local_features=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, LocalResBlock):
                x = layer(x, emb, local_features)
            else:
                x = layer(x)
        return x


class FDN(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, local_features):
        normalized = self.param_free_norm(x)
        assert local_features.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(local_features)
        beta = self.conv_beta(local_features)
        out = normalized * (1 + gamma) + beta
        return out


class LocalResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        inject_channels=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.norm_in = FDN(channels, inject_channels)
        self.norm_out = FDN(self.out_channels, inject_channels)

        self.in_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, local_conditions):
        return checkpoint(
            self._forward, (x, emb, local_conditions), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, local_conditions):
        h = self.norm_in(x, local_conditions)
        h = self.in_layers(h)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.norm_out(h, local_conditions)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h

class FeatureWarperSoftsplat(nn.Module):
    def __init__(self, with_learnable_metric=False, in_channels=128):
        super().__init__()
        self.with_learnable_metric = with_learnable_metric

        if with_learnable_metric:
            # Learn confidence (metric) from input features
            self.metric_net = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output: [B, 1, H, W]
            )

    def forward(self, feat_ref, flow):
        """
        feat_ref: tensor of shape [B, C=128, H=128, W=128]
        flow:     tensor of shape [B, 2, H, W] (optical flow in pixel units)
        """
        if self.with_learnable_metric:
            metric = self.metric_net(feat_ref)  # [B, 1, H, W]
        else:
            # Default: uniform confidence
            metric = torch.ones_like(flow[:, :1])  # shape: [B, 1, H, W]

        warped = softsplat(
            tenIn=feat_ref,
            tenFlow=flow,
            tenMetric=metric,
            strMode="soft"
        )
        return warped

class FeatureExtractorWarped(nn.Module):
    def __init__(self, inject_channels, dims=2):
        super().__init__()
        self.pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, 3, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 128, 128, 3, padding=1),
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

    def forward(self, local_conditions,flow):
        frame_features = self.pre_extractor(local_conditions,None)  # [B,C,W,H]
        warped_features = self.wrapper(frame_features, flow)

        assert len(self.extractors) == len(self.zero_convs)
        
        output_features = []
        for idx in range(len(self.extractors)):
            local_features = self.extractors[idx](warped_features, None)
            output_features.append(self.zero_convs[idx](local_features))
        return output_features

class FeatureExtractor(nn.Module):

    def __init__(self, local_channels, inject_channels, dims=2):
        super().__init__()
        self.pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, local_channels, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 64, 64, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 128, 128, 3, padding=1),
            nn.SiLU(),
        )
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
    
    def forward(self, local_conditions):
        local_features = self.pre_extractor(local_conditions, None)
        assert len(self.extractors) == len(self.zero_convs)
        
        output_features = []
        for idx in range(len(self.extractors)):
            local_features = self.extractors[idx](local_features, None)
            output_features.append(self.zero_convs[idx](local_features))
        return output_features