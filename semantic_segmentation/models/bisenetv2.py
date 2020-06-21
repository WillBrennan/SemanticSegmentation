import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return F.leaky_relu(self.bn(self.conv(x)))


class DetailBranchStage(nn.Module):
    def __init__(self, num_in: int, num_out: int, r: int):
        super().__init__()

        layers = [ConvBNRelu(num_in, num_out, kernel_size=3, padding=1, stride=2)]

        for _ in range(r):
            layers += [ConvBNRelu(num_out, num_out, kernel_size=3, padding=1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DetailBranchStage(3, 64, r=1)
        self.stage2 = DetailBranchStage(64, 128, r=2)
        self.stage3 = DetailBranchStage(128, 128, r=2)

    def forward(self, x: torch.Tensor):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, num_in: int, num_features: int):
        super().__init__()
        self.conv0 = ConvBNRelu(num_in, num_features, kernel_size=3, padding=1, stride=2)
        self.left_conv0 = ConvBNRelu(num_features, num_features // 2, kernel_size=1)
        self.left_conv1 = ConvBNRelu(num_features // 2, num_features, kernel_size=3, padding=1, stride=2)

        self.right_mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fusion = ConvBNRelu(2 * num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv0(x)
        x_left = self.left_conv0(x)
        x_left = self.left_conv1(x_left)

        x_right = self.right_mpool(x)

        x = torch.cat([x_left, x_right], dim=1)
        x = self.fusion(x)
        return x


class DWConv2DBN(nn.Module):
    def __init__(self, num_in: int, num_out: int, kernel_size: int, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=kernel_size, padding=padding, stride=stride, groups=num_in)
        self.bn = nn.BatchNorm2d(num_out)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class GatherAndExpansionLayer(nn.Module):
    def __init__(self, num_features: int, expansion_ratio: int):
        super().__init__()
        num_expanded = num_features * expansion_ratio

        self.conv = ConvBNRelu(num_features, num_features, kernel_size=3, padding=1)
        self.dwconv = DWConv2DBN(num_features, num_expanded, kernel_size=3, padding=1)
        self.conv_project = nn.Conv2d(num_expanded, num_features, kernel_size=1)
        self.bn_project = nn.BatchNorm2d(num_features)

    def forward(self, x_in: torch.Tensor):
        x = self.conv(x_in)
        x = self.dwconv(x)
        x = self.bn_project(self.conv_project(x))

        x = x + x_in
        x = F.leaky_relu(x)
        return x


class StridedGatherAndExpansionLayer(nn.Module):
    def __init__(self, num_in: int, num_out: int, stride: int, expansion_ratio: int):
        super().__init__()
        num_expanded = num_in * expansion_ratio

        self.left_conv = ConvBNRelu(num_in, num_in, kernel_size=3, padding=1)
        self.left_dwconv0 = DWConv2DBN(num_in, num_expanded, kernel_size=3, padding=1, stride=stride)
        self.left_dwconv1 = DWConv2DBN(num_expanded, num_expanded, kernel_size=3, padding=1, stride=1)
        self.left_conv_project = nn.Conv2d(num_expanded, num_out, kernel_size=1)
        self.left_bn_project = nn.BatchNorm2d(num_out)

        self.right_dwconv = DWConv2DBN(num_in, num_in, kernel_size=3, padding=1, stride=stride)
        self.right_conv_project = nn.Conv2d(num_in, num_out, kernel_size=1)
        self.right_bn_project = nn.BatchNorm2d(num_out)

    def forward(self, x: torch.Tensor):
        x_left = self.left_conv(x)
        x_left = self.left_dwconv0(x_left)
        x_left = self.left_dwconv1(x_left)
        x_left = self.left_bn_project(self.left_conv_project(x_left))

        x_right = self.right_dwconv(x)
        x_right = self.right_bn_project(self.right_conv_project(x_right))

        x = x_left + x_right
        x = F.leaky_relu(x)

        return x


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.conv_project = ConvBNRelu(num_features, num_features, kernel_size=1)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x_gap = torch.mean(x, dim=(2, 3), keepdim=True)
        x_gap = self.conv_project(x_gap)

        x = x_gap + x
        x = self.conv1(x)
        return x


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock(3, 16)

        self.stage3 = nn.Sequential(
            *[
                StridedGatherAndExpansionLayer(16, 32, stride=2, expansion_ratio=6),
                GatherAndExpansionLayer(32, expansion_ratio=6),
            ]
        )

        self.stage4 = nn.Sequential(
            *[
                StridedGatherAndExpansionLayer(32, 64, stride=2, expansion_ratio=6),
                GatherAndExpansionLayer(64, expansion_ratio=6),
            ]
        )

        self.stage5 = nn.Sequential(
            *[
                StridedGatherAndExpansionLayer(64, 128, stride=2, expansion_ratio=6),
                GatherAndExpansionLayer(128, expansion_ratio=6),
                GatherAndExpansionLayer(128, expansion_ratio=6),
                GatherAndExpansionLayer(128, expansion_ratio=6),
                ContextEmbeddingBlock(128),
            ]
        )

    def forward(self, x: torch.Tensor):
        outputs = {}
        x = self.stem(x)
        outputs['aux_c2'] = x

        x = self.stage3(x)
        outputs['aux_c3'] = x

        x = self.stage4(x)
        outputs['aux_c4'] = x

        x = self.stage5(x)
        outputs['aux_c5'] = x

        return outputs


class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        self.detail_left_dwconv0 = DWConv2DBN(num_features, num_features, kernel_size=3, padding=1)
        self.detail_left_project = nn.Conv2d(num_features, num_features, kernel_size=1)

        self.detail_right_conv0 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=2)
        self.detail_right_bn0 = nn.BatchNorm2d(num_features)
        self.detail_right_apool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.semantic_left_conv0 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.semantic_left_bn0 = nn.BatchNorm2d(num_features)

        self.semantic_right_dwconv0 = DWConv2DBN(num_features, num_features, kernel_size=3, padding=1)
        self.semantic_right_project = nn.Conv2d(num_features, num_features, kernel_size=1)

        self.fusion_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(num_features)

    def forward(self, x_semantic: torch.Tensor, x_detail: torch.Tensor):
        x_detail_left = self.detail_left_dwconv0(x_detail)
        x_detail_left = self.detail_left_project(x_detail_left)

        x_detail_right = self.detail_right_conv0(x_detail)
        x_detail_right = self.detail_right_bn0(x_detail_right)
        x_detail_right = self.detail_right_apool(x_detail_right)

        x_semantic_left = self.semantic_left_conv0(x_semantic)
        x_semantic_left = self.semantic_left_bn0(x_semantic_left)
        x_semantic_left = torch.sigmoid(x_semantic_left)
        x_semantic_left = F.interpolate(x_semantic_left, scale_factor=4, mode='bilinear', align_corners=False)

        x_semantic_right = self.semantic_right_dwconv0(x_semantic)
        x_semantic_right = self.semantic_right_project(x_semantic_right)
        x_semantic_right = torch.sigmoid(x_semantic_right)

        x_detail = x_detail_left * x_semantic_left
        x_semantic = x_detail_right * x_semantic_right

        x_semantic = F.interpolate(x_semantic, scale_factor=4, mode='bilinear', align_corners=False)

        x = x_detail + x_semantic

        x = self.fusion_bn(self.fusion_conv(x))
        return x


class SegmentationHead(nn.Module):
    def __init__(self, num_in, num_categories: int, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        # note(will.brennan):
        # doesnt specify c_t in the paper but an expansion of 6 is used elsewhere
        c_t = 6 * num_in

        self.conv0 = ConvBNRelu(num_in, c_t, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv_project = nn.Conv2d(c_t, num_categories, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.conv0(x)
        x = self.dropout(x)
        x = self.conv_project(x)

        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        return x


class BiSeNetV2(nn.Module):
    """
    note(will.brennan):
    this implements https://arxiv.org/pdf/2004.02147.pdf which is actually slow for real-time segmentation, they 
    do a slight of hand in their paper and run on half-res cityscapes and say they're faster than papers running on 
    full-res cityscapes like Fast-SCNN (which are actually 4x quicker).
    """
    def __init__(self, categories):
        super().__init__()
        logging.info(f'creating model with categories: {categories}')

        # todo(will.brennan) - find a nicer way of saving the categories in the state dict...
        self._categories = nn.ParameterDict({i: nn.Parameter(torch.Tensor(0)) for i in categories})
        num_categories = len(self._categories)

        self.semantic = SemanticBranch()
        self.detail = DetailBranch()
        self.bga = BilateralGuidedAggregationLayer(128)

        seg_head_names = ['out', 'aux_c2', 'aux_c3', 'aux_c4', 'aux_c5']
        self.seg_heads = nn.ModuleDict(
            {
                'out': SegmentationHead(128, num_categories, scale_factor=8),
                'aux_c2': SegmentationHead(16, num_categories, scale_factor=4),
                'aux_c3': SegmentationHead(32, num_categories, scale_factor=8),
                'aux_c4': SegmentationHead(64, num_categories, scale_factor=16),
                'aux_c5': SegmentationHead(128, num_categories, scale_factor=32),
            }
        )

    @property
    def categories(self):
        return self._categories

    def forward(self, x: torch.Tensor):
        x_semantic = self.semantic(x)
        x_detail = self.detail(x)

        x_bga = self.bga(x_semantic['aux_c5'], x_detail)

        # note(will.brennan) - being explicit and verbose here
        x = {
            'out': x_bga,
            'aux_c2': x_semantic['aux_c2'],
            'aux_c3': x_semantic['aux_c3'],
            'aux_c4': x_semantic['aux_c4'],
            'aux_c5': x_semantic['aux_c5'],
        }

        x = {k: self.seg_heads[k](v) for k, v in x.items()}
        return x
