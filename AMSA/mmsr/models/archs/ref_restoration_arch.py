import mmsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsr.models.archs.DCNv2.dcn_v2 import DCN_sep_pre_multi_offset as DynAgg
from mmsr.models.archs.DCNv2.dcn_v2 import DCN_pre_offset as DynWarp


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


class RestorationNet(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(
            ngf, n_blocks, groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.small_deform_conv.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_deform_conv.conv_offset_mask.bias.data.zero_()

        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_deform_conv.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.medium_deform_conv.conv_offset_mask.bias.data.zero_()

        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_deform_conv.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_deform_conv.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset_list, img_ref_list, max_val_list):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """
        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset_list,
                                               img_ref_list, max_val_list)
        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(DynamicAggregationRestoration, self).__init__()

        # dynamic aggregation module for relu3_1 reference feature
        self.small_deform_conv = DynWarp(
            256,
            256,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
            extra_offset_mask=False
        )
        self.small_offset_conv1 = nn.Conv2d(
            ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(
            256,
            256,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for small scale restoration
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_small = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu2_1 reference feature
        self.medium_deform_conv = DynWarp(
            128,
            128,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
            extra_offset_mask=False
        )
        self.medium_offset_conv1 = nn.Conv2d(
            ngf + 128, 128, 3, 1, 1, bias=True)
        self.medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.medium_dyn_agg = DynAgg(
            128,
            128,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for medium scale restoration
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_medium = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu1_1 reference feature
        self.large_deform_conv = DynWarp(
            64,
            64,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
            extra_offset_mask=False
        )
        self.large_offset_conv1 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        self.large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.large_dyn_agg = DynAgg(
            64,
            64,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for large scale
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_large = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, pre_offset_list, img_ref_list, max_val_list):
        # dynamic aggregation for relu3_1 reference feature
        # max_val [b,1,h,w]
        max_val = torch.cat(max_val_list, dim=1)
        max_val = torch.softmax(max_val * 10, dim=1)
        input = x
        for i in range(len(img_ref_list)):
            x = input
            img_ref_feat_relu31 = self.small_deform_conv(img_ref_list[i]['relu3_1'], pre_offset_list[i]['relu3_1'])
            relu3_offset = torch.cat([x, img_ref_feat_relu31], 1)
            relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset))
            relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset))
            relu3_swapped_feat = self.lrelu(
                self.small_dyn_agg([img_ref_list[i]['relu3_1'], relu3_offset],
                                   pre_offset_list[i]['relu3_1']))
            # small scale
            h = torch.cat([x, relu3_swapped_feat], 1)
            h = self.head_small(h)
            h = self.body_small(h) + x
            x = self.tail_small(h)

            # dynamic aggregation for relu2_1 reference feature
            img_ref_feat_relu21 = self.medium_deform_conv(img_ref_list[i]['relu2_1'], pre_offset_list[i]['relu2_1'])
            relu2_offset = torch.cat([x, img_ref_feat_relu21], 1)
            relu2_offset = self.lrelu(self.medium_offset_conv1(relu2_offset))
            relu2_offset = self.lrelu(self.medium_offset_conv2(relu2_offset))
            relu2_swapped_feat = self.lrelu(
                self.medium_dyn_agg([img_ref_list[i]['relu2_1'], relu2_offset],
                                    pre_offset_list[i]['relu2_1']))
            # medium scale
            h = torch.cat([x, relu2_swapped_feat], 1)
            h = self.head_medium(h)
            h = self.body_medium(h) + x
            x = self.tail_medium(h)

            # dynamic aggregation for relu1_1 reference feature
            img_ref_feat_relu11 = self.large_deform_conv(img_ref_list[i]['relu1_1'], pre_offset_list[i]['relu1_1'])
            relu1_offset = torch.cat([x, img_ref_feat_relu11], 1)
            relu1_offset = self.lrelu(self.large_offset_conv1(relu1_offset))
            relu1_offset = self.lrelu(self.large_offset_conv2(relu1_offset))
            relu1_swapped_feat = self.lrelu(
                self.large_dyn_agg([img_ref_list[i]['relu1_1'], relu1_offset],
                                   pre_offset_list[i]['relu1_1']))
            # large scale

            h = torch.cat([x, relu1_swapped_feat], 1)
            h = self.head_large(h)
            h = self.body_large(h) + x
            if i == 0:
                h0 = h * max_val[:, i, :, :].unsqueeze(1)
            else:
                h0 += h * max_val[:, i, :, :].unsqueeze(1)

        x = self.tail_large(h0)

        return x
