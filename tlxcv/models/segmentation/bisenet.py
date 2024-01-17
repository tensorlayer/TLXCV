import tensorlayerx as tlx
import tensorlayerx.nn as nn
from .layers import Activation, Add, ConvBN, ConvBNReLU, DepthwiseConvBN

__all__ = ["BiSeNetV2"]


class BiSeNetV2(nn.Module):
    """
    The BiSeNet V2 implementation based on TensorlayerX.

    The original article refers to
    Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
    (https://arxiv.org/abs/2004.02147)

    Args:
        num_classes (int): The unique number of target classes.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
    """

    def __init__(
        self, num_classes, lambd=0.25, align_corners=False, data_format="channels_first"
    ):
        super().__init__()
        self.data_format = data_format
        C1, C2, C3 = 64, 64, 128
        db_channels = C1, C2, C3
        C1, C3, C4, C5 = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = C1, C3, C4, C5
        mid_channels = 128
        self.db = DetailBranch(db_channels, data_format=data_format)
        self.sb = SemanticBranch(sb_channels, data_format=data_format)
        self.bga = BGA(mid_channels, align_corners, data_format=data_format)
        self.aux_head1 = SegHead(C1, C1, num_classes, data_format=data_format)
        self.aux_head2 = SegHead(C3, C3, num_classes, data_format=data_format)
        self.aux_head3 = SegHead(C4, C4, num_classes, data_format=data_format)
        self.aux_head4 = SegHead(C5, C5, num_classes, data_format=data_format)
        self.head = SegHead(
            mid_channels, mid_channels, num_classes, data_format=data_format
        )
        self.align_corners = align_corners

    def forward(self, x):
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        logit = self.head(self.bga(dfm, sfm))
        if not self.is_train:
            logit_list = [logit]
        else:
            logit1 = self.aux_head1(feat1)
            logit2 = self.aux_head2(feat2)
            logit3 = self.aux_head3(feat3)
            logit4 = self.aux_head4(feat4)
            logit_list = [logit, logit1, logit2, logit3, logit4]
        if self.data_format == "channels_first":
            logit_sizes = [tlx.get_tensor_shape(
                logit)[2:] for logit in logit_list]
            x_size = tlx.get_tensor_shape(x)[2:]
        else:
            logit_sizes = [tlx.get_tensor_shape(
                logit)[1:3] for logit in logit_list]
            x_size = tlx.get_tensor_shape(x)[1:3]
        scales = [
            (x_size[0] / l_size[0], x_size[1] / l_size[1]) for l_size in logit_sizes
        ]
        logit_list = [
            tlx.Resize(
                scale=scale,
                method="bilinear",
                antialias=self.align_corners,
                data_format=self.data_format,
            )(logit)
            for logit, scale in zip(logit_list, scales)
        ]
        return logit_list[0]


class StemBlock(nn.Module):
    def __init__(self, in_dim, out_dim, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        self.conv = ConvBNReLU(
            in_dim, out_dim, 3, stride=2, data_format=data_format)
        self.left = nn.Sequential(
            [
                ConvBNReLU(out_dim, out_dim // 2, 1, data_format=data_format),
                ConvBNReLU(out_dim // 2, out_dim, 3,
                           stride=2, data_format=data_format),
            ]
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, data_format=data_format
        )
        self.fuse = ConvBNReLU(out_dim * 2, out_dim, 3,
                               data_format=data_format)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        concat = tlx.concat(
            [left, right], axis=1 if self.data_format == "channels_first" else -1
        )
        return self.fuse(concat)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, data_format="channels_first"):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1, data_format=data_format)
        self.bn = nn.BatchNorm2d(num_features=in_dim, data_format=data_format)
        self.conv_1x1 = ConvBNReLU(in_dim, out_dim, 1, data_format=data_format)
        self.add = Add()
        self.conv_3x3 = nn.GroupConv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            data_format=data_format,
        )

    def forward(self, x):
        gap = self.gap(x)
        bn = self.bn(gap)
        conv1 = self.add(self.conv_1x1(bn), x)
        return self.conv_3x3(conv1)


class GatherAndExpansionLayer1(nn.Module):
    """Gather And Expansion Layer with stride 1"""

    def __init__(self, in_dim, out_dim, expand, data_format="channels_first"):
        super().__init__()
        expand_dim = expand * in_dim
        self.conv = nn.Sequential(
            [
                ConvBNReLU(in_dim, in_dim, 3, data_format=data_format),
                DepthwiseConvBN(in_dim, expand_dim, 3,
                                data_format=data_format),
                ConvBN(expand_dim, out_dim, 1, data_format=data_format),
            ]
        )
        self.relu = Activation("relu")

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class GatherAndExpansionLayer2(nn.Module):
    """Gather And Expansion Layer with stride 2"""

    def __init__(self, in_dim, out_dim, expand, data_format="channels_first"):
        super().__init__()
        expand_dim = expand * in_dim
        self.branch_1 = nn.Sequential(
            [
                ConvBNReLU(in_dim, in_dim, 3, data_format=data_format),
                DepthwiseConvBN(
                    in_dim, expand_dim, 3, stride=2, data_format=data_format
                ),
                DepthwiseConvBN(expand_dim, expand_dim, 3,
                                data_format=data_format),
                ConvBN(expand_dim, out_dim, 1, data_format=data_format),
            ]
        )
        self.branch_2 = nn.Sequential(
            [
                DepthwiseConvBN(in_dim, in_dim, 3, stride=2,
                                data_format=data_format),
                ConvBN(in_dim, out_dim, 1, data_format=data_format),
            ]
        )
        self.relu = Activation("relu")

    def forward(self, x):
        return self.relu(self.branch_1(x) + self.branch_2(x))


class DetailBranch(nn.Module):
    """The detail branch of BiSeNet, which has wide channels but shallow layers."""

    def __init__(self, in_channels, data_format="channels_first"):
        super().__init__()
        C1, C2, C3 = in_channels
        self.convs = nn.Sequential(
            [
                ConvBNReLU(3, C1, 3, stride=2, data_format=data_format),
                ConvBNReLU(C1, C1, 3, data_format=data_format),
                ConvBNReLU(C1, C2, 3, stride=2, data_format=data_format),
                ConvBNReLU(C2, C2, 3, data_format=data_format),
                ConvBNReLU(C2, C2, 3, data_format=data_format),
                ConvBNReLU(C2, C3, 3, stride=2, data_format=data_format),
                ConvBNReLU(C3, C3, 3, data_format=data_format),
                ConvBNReLU(C3, C3, 3, data_format=data_format),
            ]
        )

    def forward(self, x):
        return self.convs(x)


class SemanticBranch(nn.Module):
    """The semantic branch of BiSeNet, which has narrow channels but deep layers."""

    def __init__(self, in_channels, data_format="channels_first"):
        super().__init__()
        C1, C3, C4, C5 = in_channels
        self.stem = StemBlock(3, C1, data_format=data_format)
        self.stage3 = nn.Sequential(
            [
                GatherAndExpansionLayer2(C1, C3, 6, data_format=data_format),
                GatherAndExpansionLayer1(C3, C3, 6, data_format=data_format),
            ]
        )
        self.stage4 = nn.Sequential(
            [
                GatherAndExpansionLayer2(C3, C4, 6, data_format=data_format),
                GatherAndExpansionLayer1(C4, C4, 6, data_format=data_format),
            ]
        )
        self.stage5_4 = nn.Sequential(
            [
                GatherAndExpansionLayer2(C4, C5, 6, data_format=data_format),
                GatherAndExpansionLayer1(C5, C5, 6, data_format=data_format),
                GatherAndExpansionLayer1(C5, C5, 6, data_format=data_format),
                GatherAndExpansionLayer1(C5, C5, 6, data_format=data_format),
            ]
        )
        self.ce = ContextEmbeddingBlock(C5, C5, data_format=data_format)

    def forward(self, x):
        stage2 = self.stem(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5_4 = self.stage5_4(stage4)
        fm = self.ce(stage5_4)
        return stage2, stage3, stage4, stage5_4, fm


class BGA(nn.Module):
    """The Bilateral Guided Aggregation Layer, used to fuse the semantic features and spatial features."""

    def __init__(self, out_dim, align_corners, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        self.align_corners = align_corners
        self.db_branch_keep = nn.Sequential(
            [
                DepthwiseConvBN(out_dim, out_dim, 3, data_format=data_format),
                nn.GroupConv2d(
                    in_channels=out_dim,
                    out_channels=out_dim,
                    kernel_size=1,
                    padding=0,
                    data_format=data_format,
                ),
            ]
        )
        self.db_branch_down = nn.Sequential(
            [
                ConvBN(out_dim, out_dim, 3, stride=2, data_format=data_format),
                nn.AvgPool2d(
                    kernel_size=3, stride=2, padding=1, data_format=data_format
                ),
            ]
        )
        self.sb_branch_keep = nn.Sequential(
            [
                DepthwiseConvBN(out_dim, out_dim, 3, data_format=data_format),
                nn.GroupConv2d(
                    in_channels=out_dim,
                    out_channels=out_dim,
                    kernel_size=1,
                    padding=0,
                    data_format=data_format,
                ),
                Activation(act="sigmoid"),
            ]
        )
        self.sb_branch_up = ConvBN(
            out_dim, out_dim, 3, data_format=data_format)
        self.conv = ConvBN(out_dim, out_dim, 3, data_format=data_format)

    def forward(self, dfm, sfm):
        db_feat_keep = self.db_branch_keep(dfm)
        db_feat_down = self.db_branch_down(dfm)
        sb_feat_keep = self.sb_branch_keep(sfm)
        sb_feat_up = self.sb_branch_up(sfm)
        if self.data_format == "channels_first":
            in_size = tlx.get_tensor_shape(sb_feat_up)[2:]
            out_size = tlx.get_tensor_shape(db_feat_keep)[2:]
        else:
            in_size = tlx.get_tensor_shape(sb_feat_up)[1:3]
            out_size = tlx.get_tensor_shape(db_feat_keep)[1:3]
        scale = (out_size[0] / in_size[0], out_size[1] / in_size[1])
        sb_feat_up = tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(sb_feat_up)
        sb_feat_up = tlx.sigmoid(sb_feat_up)
        db_feat = db_feat_keep * sb_feat_up
        sb_feat = db_feat_down * sb_feat_keep
        if self.data_format == "channels_first":
            in_size = tlx.get_tensor_shape(sb_feat)[2:]
            out_size = tlx.get_tensor_shape(db_feat)[2:]
        else:
            in_size = tlx.get_tensor_shape(sb_feat)[1:3]
            out_size = tlx.get_tensor_shape(db_feat)[1:3]
        scale = (out_size[0] / in_size[0], out_size[1] / in_size[1])
        sb_feat = tlx.Resize(
            scale=scale,
            method="bilinear",
            antialias=self.align_corners,
            data_format=self.data_format,
        )(sb_feat)
        return self.conv(db_feat + sb_feat)


class SegHead(nn.Module):
    def __init__(self, in_dim, mid_dim, num_classes, data_format="channels_first"):
        super().__init__()
        self.conv_3x3 = nn.Sequential(
            [
                ConvBNReLU(in_dim, mid_dim, 3, data_format=data_format),
                nn.Dropout(0.1),
            ]
        )
        self.conv_1x1 = nn.GroupConv2d(
            in_channels=mid_dim,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            data_format=data_format,
        )

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        return conv2
