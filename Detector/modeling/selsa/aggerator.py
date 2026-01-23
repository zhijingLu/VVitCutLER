import torch
import torch.nn as nn

from detectron2.utils.registry import Registry

AGGREGATOR_REGISTRY = Registry("Aggregator")

def build_aggregator(cfg):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.AGGREGATOR.NAME

    return AGGREGATOR_REGISTRY.get(name)(cfg)

@AGGREGATOR_REGISTRY.register()
class SelsaAggregator(nn.Module):
    """Selsa aggregator module.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        in_channels (int): The number of channels of the features of
            proposal.
        num_attention_blocks (int): The number of attention blocks used in
            selsa aggregator module. Defaults to 16.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, in_channels, num_attention_blocks=16):
        super().__init__()
        self.fc_embed = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True)  # 新增激活层
        )
        self.ref_fc_embed = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True)  # 新增激活层
        )
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks
        self.norm = nn.LayerNorm(in_channels)  # 新增归一化层


    def forward(self, x, ref_x):
        """Aggregate the features `ref_x` of reference proposals.

        The aggregation mainly contains two steps:
        1. Use multi-head attention to computing the weight between `x` and
        `ref_x`.
        2. Use the normlized (i.e. softmax) weight to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [N, C]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C]. M is the number of reference frame
                proposals.

        Returns:
            Tensor: The aggregated features of key frame proposals with shape
            [N, C].
        """

        roi_n, C = x.shape
        ref_roi_n, _ = ref_x.shape
        num_c_per_att_block = C // self.num_attention_blocks
        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               num_c_per_att_block).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       num_c_per_att_block).permute(1, 2, 0)

        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   num_c_per_att_block).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, C))
        x_new = self.norm(x + x_new) 
        return x_new