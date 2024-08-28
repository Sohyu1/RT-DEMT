import copy as cp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, Linear,
                      build_activation_layer, build_conv_layer,
                      build_norm_layer, build_upsample_layer)

from .utils import get_activation


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x


class TopDownMultiStageHead(nn.Module):
    """Heads for multi-stage pose models.

    TopDownMultiStageHead is consisted of multiple branches, each of
    which has num_deconv_layers(>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_stages (int): Number of stages.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=17,
                 num_stages=1,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_stages = num_stages

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        # build multi-stage deconv layers
        self.multi_deconv_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            if num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(
                    num_deconv_layers,
                    num_deconv_filters,
                    num_deconv_kernels,
                )
            elif num_deconv_layers == 0:
                deconv_layers = nn.Identity()
            else:
                raise ValueError(
                    f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
            self.multi_deconv_layers.append(deconv_layers)

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        # build multi-stage final layers
        self.multi_final_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            if identity_final_layer:
                final_layer = nn.Identity()
            else:
                final_layer = build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=num_deconv_filters[-1]
                    if num_deconv_layers > 0 else in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding)
            self.multi_final_layers.append(final_layer)

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []
        assert isinstance(x, list)
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)
        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding


class PredictHeatmap(nn.Module):
    """Predict the heat map for an input feature.

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        use_prm (bool): Whether to use pose refine machine. Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 unit_channels,
                 out_channels,
                 out_shape,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.use_prm = use_prm
        if use_prm:
            self.prm = PRM(out_channels, norm_cfg=norm_cfg)
        self.conv_layers = nn.Sequential(
            ConvModule(
                unit_channels,
                unit_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=False),
            ConvModule(
                unit_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False))

    def forward(self, feature):
        feature = self.conv_layers(feature)
        output = nn.functional.interpolate(
            feature, size=self.out_shape, mode='bilinear', align_corners=True)
        if self.use_prm:
            output = self.prm(output)
        return output


class PRM(nn.Module):
    """Pose Refine Machine.

    For more details about PRM, refer to Learning Delicate
    Local Representations for Multi-Person Pose Estimation (ECCV 2020).
    Args:
        out_channels (int): Channel number of the output. Equals to
            the number of key points.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, out_channels, norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.out_channels = out_channels
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_path = nn.Sequential(
            Linear(self.out_channels, self.out_channels),
            build_norm_layer(dict(type='BN1d'), out_channels)[1],
            build_activation_layer(dict(type='ReLU')),
            Linear(self.out_channels, self.out_channels),
            build_norm_layer(dict(type='BN1d'), out_channels)[1],
            build_activation_layer(dict(type='ReLU')),
            build_activation_layer(dict(type='Sigmoid')))

        self.bottom_path = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=False),
            DepthwiseSeparableConvModule(
                self.out_channels,
                1,
                kernel_size=9,
                stride=1,
                padding=4,
                norm_cfg=norm_cfg,
                inplace=False), build_activation_layer(dict(type='Sigmoid')))
        self.conv_bn_relu_prm_1 = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False)

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out

        out_2 = self.global_pooling(out_1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_2 = self.middle_path(out_2)
        out_2 = out_2.unsqueeze(2)
        out_2 = out_2.unsqueeze(3)

        out_3 = self.bottom_path(out_1)
        out = out_1 * (1 + out_2 * out_3)

        return out


class TopDownMSMUHead(nn.Module):
    """Heads for multi-stage multi-unit heads used in Multi-Stage Pose
    estimation Network (MSPN), and Residual Steps Networks (RSN).

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        num_stages (int): Number of stages.
        num_units (int): Number of units in each stage.
        use_prm (bool): Whether to use pose refine machine (PRM).
            Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 out_shape,
                 unit_channels=256,
                 out_channels=17,
                 num_stages=4,
                 num_units=4,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()

        self.out_shape = out_shape
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.predict_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            for j in range(self.num_units):
                self.predict_layers.append(
                    PredictHeatmap(
                        unit_channels,
                        out_channels,
                        out_shape,
                        use_prm,
                        norm_cfg=norm_cfg))

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages
                                and units.
        """
        out = []
        assert isinstance(x, list)
        assert len(x) == self.num_stages
        assert isinstance(x[0], list)
        assert len(x[0]) == self.num_units
        assert x[0][0].shape[1] == self.unit_channels
        for i in range(self.num_stages):
            for j in range(self.num_units):
                y = self.predict_layers[i * self.num_units + j](x[i][j])
                out.append(y)

        return out


class IntegralPoseRegressionHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 feat_size,
                 out_sigma=False,
                 debias=False,):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.out_sigma = out_sigma
        self.debias = debias

        self.layer = DeconvHead(in_channels=in_channels, num_layers=3, num_filters=256, kernel_size=4,
                                conv_kernel_size=1, num_joints=num_joints, depth_dim=1)

        self.size = feat_size
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1]) / self.size
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size]) / self.size
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

        if out_sigma:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.in_channels, self.num_joints * 2)
        if debias:
            self.softmax_fc = nn.Linear(128 * 128, 84)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, ('DeepPoseRegressionHead only supports '
                                 'single-level feature.')
            x = x[0]

        featmap = self.layer(x)
        s = list(featmap.size())
        featmap = featmap.view([s[0], s[1], s[2] * s[3]])
        if self.debias:
            mlp_x_norm = torch.norm(self.softmax_fc.weight, dim=-1)
            norm_feat = torch.norm(featmap, dim=-1, keepdim=True)
            featmap = self.softmax_fc(featmap)
            featmap /= norm_feat
            featmap /= mlp_x_norm.reshape(1, 1, -1)

        featmap = F.softmax(16 * featmap, dim=2)
        featmap = featmap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = featmap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2, keepdim=True)
        scoremap_y = featmap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2, keepdim=True)
        # output = torch.cat([soft_argmax_x, soft_argmax_y], dim=-1)

        if self.debias:
            C = featmap.reshape(s[0], s[1], s[2] * s[3]).exp().sum(dim=2).unsqueeze(dim=2)
            soft_argmax_x = C / (C - 1) * (soft_argmax_x - 1 / (2 * C))
            soft_argmax_y = C / (C - 1) * (soft_argmax_y - 1 / (2 * C))

        output = torch.cat([soft_argmax_x, soft_argmax_y], dim=-1)
        if self.out_sigma:
            x = self.gap(x).reshape(x.size(0), -1)
            pred_sigma = self.fc(x)
            pred_sigma = pred_sigma.reshape(pred_sigma.size(0), self.num_joints, 2)
            output = torch.cat([output, pred_sigma], dim=-1)

        return output, featmap


class SimCCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 feat_size,
                 img_size,
                 simcc_ratio):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.img_size = img_size
        self.simcc_ratio = simcc_ratio

        self.dconvlayer = DeconvHead(in_channels=in_channels, num_layers=3, num_filters=256, kernel_size=4,
                                     conv_kernel_size=3, num_joints=num_joints, depth_dim=1)

        self.final_layer = nn.Conv2d(
            in_channels=self.num_joints,
            out_channels=self.num_joints,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.W = int(self.img_size[0] * self.simcc_ratio)
        self.H = int(self.img_size[1] * self.simcc_ratio)

        self.mlp_head_x = nn.Linear(feat_size ** 2, self.W)
        self.mlp_head_y = nn.Linear(feat_size ** 2, self.H)

    def forward(self, x):
        """Forward function."""
        if self.dconvlayer is None:
            feats = x[-1]
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.dconvlayer(x)

        x = feats.flatten(2)
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        output = self.get_loc(pred_x, pred_y)

        return output

    def get_loc(self, pred_x, pred_y):
        # 直接argmax梯度会消失
        # loc_x = torch.argmax(pred_x, dim=2) / self.W
        # loc_y = torch.argmax(pred_y, dim=2) / self.H
        gates_onehot_x = F.gumbel_softmax(pred_x, tau=0.1, hard=True, dim=2)
        gates_onehot_y = F.gumbel_softmax(pred_y, tau=0.1, hard=True, dim=2)
        device = gates_onehot_x.device
        loc_x = torch.mul(gates_onehot_x, torch.arange(1280).to(device)).sum(dim=2) / self.W
        loc_y = torch.mul(gates_onehot_y, torch.arange(1280).to(device)).sum(dim=2) / self.H

        locs = torch.stack((loc_x, loc_y), dim=-1)
        val_x = torch.amax(pred_x, dim=2)
        val_y = torch.amax(pred_y, dim=2)

        mask = val_x > val_y
        val_x[mask] = val_y[mask]
        val = val_x
        locs[val <= 0.] = 0

        return locs.flatten(1)


class HeatMapHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 feat_size):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.dconvlayer = DeconvHead(in_channels=in_channels, num_layers=3, num_filters=256, kernel_size=4,
                                     conv_kernel_size=1, num_joints=num_joints, depth_dim=1)

        self.final_layer = nn.Conv2d(
            in_channels=self.num_joints,
            out_channels=self.num_joints,
            kernel_size=3,
            stride=1,
            padding=1
        )


    def forward(self, x):
        """Forward function."""
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, ('DeepPoseRegressionHead only supports '
                                 'single-level feature.')
            x = x[0]

        featmap = self.dconvlayer(x)
        featmap = self.final_layer(featmap)

        pred_x = torch.argmax(featmap, dim=1)
        _, pred_y = torch.max(featmap, dim=3)

        output = torch.cat([pred_x.unsqueeze(2), pred_y.unsqueeze(2)], dim=2) / 640
        output = output.flatten(1)

        return output, featmap


class IRPHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 feat_size,
                 out_sigma=False,
                 debias=False,):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.out_sigma = out_sigma
        self.debias = debias

        self.layer = DeconvHead(in_channels=in_channels, num_layers=3, num_filters=256, kernel_size=4,
                                conv_kernel_size=1, num_joints=num_joints, depth_dim=1)

        self.size = feat_size
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1]) / self.size
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size]) / self.size
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

        if out_sigma:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.in_channels, self.num_joints * 2)

    def forward(self, x):
        """Forward function."""
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, ('DeepPoseRegressionHead only supports '
                                 'single-level feature.')
            x = x[0]

        featmap = self.layer(x)
        s = list(featmap.size())
        featmap = featmap.view([s[0], s[1], s[2] * s[3]])
        featmap = F.softmax(16 * featmap, dim=2)
        featmap = featmap.view([s[0], s[1], s[2], s[3]])

        scoremap_x = featmap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2, keepdim=True)
        scoremap_y = featmap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2, keepdim=True)

        if self.debias:
            C = featmap.reshape(s[0], s[1], s[2] * s[3]).exp().sum(dim=2).unsqueeze(dim=2)
            soft_argmax_x = C / (C - 1) * (soft_argmax_x - 1 / (2 * C))
            soft_argmax_y = C / (C - 1) * (soft_argmax_y - 1 / (2 * C))

        output = torch.cat([soft_argmax_x, soft_argmax_y], dim=-1)
        # if self.out_sigma:
        #     x = self.gap(x).reshape(x.size(0), -1)
        #     pred_sigma = self.fc(x)
        #     pred_sigma = pred_sigma.reshape(pred_sigma.size(0), self.num_joints, 2)
        #     output = torch.cat([output, pred_sigma], dim=-1)

        return output, featmap


# me1 = torch.rand((8, 300, 256))
# me3 = torch.rand((8, 300, 16, 16))
# # # # me = [me1, me2, me3]
# head = SimCCHead(in_channels=300, num_joints=84, feat_size=32 * (2 ** 2))
# out, featmap = head(me3)
# head = TopDownMSMUHead(out_shape=(640, 640), unit_channels=500, out_channels=16, num_stages=3, num_units=1)
# out = head(me)
#
# import numpy as np
# head = TopDownMultiStageHead(in_channels=500)
# out = head(me)
