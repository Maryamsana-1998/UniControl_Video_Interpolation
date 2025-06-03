import torch
import torch as th
import torch.nn as nn
import omegaconf 
from ldm.util import exists
from models.softsplat import softsplat
from models.adapter_utils import *
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from src.train.util import adaptive_weighted_downsample , normalize_for_warping
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import  ResBlock, Downsample, AttentionBlock
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def resize_and_normalize_flow_batched(flow_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize and normalize batched 2D optical flow tensors for warping.

    Args:
        flow_tensor (torch.Tensor): Flow tensor of shape [B, 2, H, W].
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        torch.Tensor: Normalized flow tensor of shape [B, 2, target_h, target_w].
    """
    # Resize with bilinear interpolation
    resized = F.interpolate(flow_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    # Prepare normalization grid
    norm_w = (target_w - 1) / 2.0
    norm_h = (target_h - 1) / 2.0

    # Normalize u and v components
    u = resized[:, 0] / norm_w  # [B, target_h, target_w]
    v = resized[:, 1] / norm_h  # [B, target_h, target_w]
    normalized = torch.stack([u, v], dim=1)  # [B, 2, target_h, target_w]

    return normalized


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
        self.extractors_first = nn.ModuleList([
            LocalTimestepEmbedSequential(conv_nd(dims, 64, int(inject_channels[0] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[0] / 2), int(inject_channels[1] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[1] / 2), int(inject_channels[2] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[2] / 2), int(inject_channels[3] / 2), 3, padding=1, stride=2), nn.SiLU())
        ])

        self.extractors_last = nn.ModuleList([
            LocalTimestepEmbedSequential(conv_nd(dims, 64, int(inject_channels[0] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[0] / 2), int(inject_channels[1] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[1] / 2), int(inject_channels[2] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[2] / 2), int(inject_channels[3] / 2), 3, padding=1, stride=2), nn.SiLU())
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

        # print('print shapes:',first_frame.shape, last_frame.shape, flow_fwd.shape,flow_bwd.shape)

        first_features = self.first_pre_extractor(first_frame,None)
        last_features = self.last_pre_extractor(last_frame,None)
        
        assert len(self.extractors_first) == len(self.zero_convs) == len(self.extractors_last)
        output_features = []

        # normalize and interpolate 
        flow_res = [64,32,16, 8]

        for idx in range(len(self.extractors_first)):

            first_features = self.extractors_first[idx](first_features, None)
            last_features = self.extractors_last[idx](last_features, None)

            flow = resize_and_normalize_flow_batched(flow_fwd, target_h=flow_res[idx], target_w=flow_res[idx])
            flow_b = resize_and_normalize_flow_batched(flow_bwd, target_h=flow_res[idx], target_w=flow_res[idx])

            wrapped_first = self.wrapper(first_features, flow)
            wrapped_last = self.wrapper(last_features,flow_b)

            local_features = torch.cat([wrapped_first,wrapped_last],dim=1)
            # print('local shape', local_features.shape)

            output_feature = self.zero_convs[idx](local_features)
            output_features.append(output_feature)
        return output_features
    
class LocalAdapter(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            inject_channels,
            inject_layers,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.inject_layers = inject_layers
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.feature_extractor = Bi_Dir_FeatureExtractor(inject_channels)
        self.input_blocks = nn.ModuleList(
            [
                LocalTimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                if (1 + 3*level + nr) in self.inject_layers:
                    layers = [
                        LocalResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            inject_channels=inject_channels[level]
                        )
                    ]
                else:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(LocalTimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    LocalTimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = LocalTimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return LocalTimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, timesteps, context, local_conditions, flow, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        local_features = self.feature_extractor(local_conditions,flow)

        outs = []
        h = x.type(self.dtype)
        for layer_idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if layer_idx in self.inject_layers:
                h = module(h, emb, context, local_features[self.inject_layers.index(layer_idx)])
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
