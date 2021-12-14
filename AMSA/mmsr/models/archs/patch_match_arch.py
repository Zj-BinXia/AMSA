import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class DeformSearch(nn.Module):
    def __init__(self, N=9, kernel_size=3):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformSearch, self).__init__()
        self.N = N
        y, x = torch.meshgrid((
            torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1),
            torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1)))
        yoffset = y.flatten().view(1, 1, self.N, 1, 1)
        xoffset = x.flatten().view(1, 1, self.N, 1, 1)
        self.register_buffer('yoffset', yoffset)
        self.register_buffer('xoffset', xoffset)

    def forward(self, x, inref_y, inref_x):
        # inref # [B,K,1,H,W] x[B,CN,H2,W2],
        # K represents the number of sample points，N is surrounding point for each sample point. x is the image for sample
        b, k, _, h, w = inref_y.size()
        _, c, h1, w1 = x.size()
        inref = inref_x + inref_y * w1  # [B,K,1,H,W]
        inref = inref.view(b, 1, -1).expand(-1, c, -1).long()  # [B,C,KHW]
        recon = torch.gather(x.view(b, c, -1), dim=2, index=inref).view(b, c, k, -1).permute(0, 2, 1, 3)  # [B,C,K,H*W]
        return recon


class Evaluate(nn.Module):
    def __init__(self, filter_size):
        super(Evaluate, self).__init__()
        self.filter_size = filter_size
        self.DeformSearch = DeformSearch()

    def forward(self, input_features, ref_features, aggregated_x, aggregated_y, is_test=False):
        # input  [B,C*N,H,W] ref [B,CN,H,W] aggx  [B,K,H,W]
        b, cn, g, w = input_features.size()
        input_features = input_features.view(b, 1, cn, -1).expand(-1, aggregated_x.size()[1], -1, -1)  # [B,K=9,C*N,H*W]
        right_x_coordinate = aggregated_x.unsqueeze(2)  # [B,K,1,H,W]
        right_y_coordinate = aggregated_y.unsqueeze(2)

        ref = self.DeformSearch(ref_features, right_y_coordinate, right_x_coordinate)  # [B,K=9,C*N,H*W]
        S = torch.einsum('btfh,btfh->bth', ref, input_features)  # [B,K=9,H*W]
        # measure similarity between ref and LR image by inner product，[B*9,H,W]

        S = S.view(input_features.size()[0],
                   S.size()[1],
                   aggregated_x.size()[2],
                   aggregated_x.size()[3])  # [B,K=9,H,W]
        if is_test:
            return S
        S = torch.argmax(S, dim=1).unsqueeze(1)  # [B,1,H,W]
        offset_x = torch.gather(aggregated_x, index=S, dim=1)  # [B,1,H,W]
        offset_y = torch.gather(aggregated_y, index=S, dim=1)

        return offset_x, offset_y


class Propagation(nn.Module):
    def __init__(self, filter_size):
        super(Propagation, self).__init__()
        self.filter_size = filter_size
        label = torch.arange(0, self.filter_size * self.filter_size).repeat(self.filter_size * self.filter_size).view(
            self.filter_size * self.filter_size, 1, 1, self.filter_size, self.filter_size)  # [9,1,1,3,3]

        self.one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float().to("cuda:0")  ##[9,1,1,3,3]

    def forward(self, inref_x, inref_y, dilation):
        # inref #[B,1,H,W]
        dilation = int(dilation)
        pad_size = dilation
        inref_x = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))(inref_x)
        inref_y = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))(inref_y)  # 填充

        inref_x = inref_x.view(inref_x.size()[0], 1, inref_x.size()[1], inref_x.size()[2],
                               inref_x.size()[3])  # [B,1,1,H+2p,W+2p]
        inref_y = inref_y.view(inref_y.size()[0], 1, inref_y.size()[1], inref_y.size()[2],
                               inref_y.size()[3])  # [B,1,1,H+2p,W+2p]

        aggregated_x = F.conv3d(inref_x, self.one_hot_filter, padding=0, dilation=(1, dilation, dilation))
        # [B,K=9,1,H,W],the K=9 dimension indicates the surrounding 8 points and the center point
        aggregated_y = F.conv3d(inref_y, self.one_hot_filter, padding=0, dilation=(1, dilation, dilation))
        # [B,K=9,1,H,W],the K=9 dimension indicates the surrounding 8 points and the center point
        aggregated_x = aggregated_x.view(  # [B,K=9,H,W]
            aggregated_x.size()[0],
            aggregated_x.size()[1] * aggregated_x.size()[2],
            aggregated_x.size()[3],
            aggregated_x.size()[4])

        aggregated_y = aggregated_y.view(  # [B,K=9,H,W]
            aggregated_y.size()[0],
            aggregated_y.size()[1] * aggregated_y.size()[2],
            aggregated_y.size()[3],
            aggregated_y.size()[4])

        return aggregated_x, aggregated_y


class PatchMatch(nn.Module):
    def __init__(self):
        super(PatchMatch, self).__init__()
        self.propagation_filter_size = 3  # patch_match_args.propagation_filter_size#3
        self.alpha = 0.5
        self.propagation = Propagation(self.propagation_filter_size)
        self.evaluate = Evaluate(self.propagation_filter_size)
        N = 9
        kernel_size = 3
        y, x = torch.meshgrid((
            torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1),
            torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1)))
        yoffset = y.flatten().view(1, N, 1, 1)
        xoffset = x.flatten().view(1, N, 1, 1)
        dx = torch.Tensor([0, -1, -1, 0, 0, 0, 1, 1, 0]).view(1, N, 1, 1)
        dy = torch.Tensor([1, 0, 0, 1, 0, -1, 0, 0, -1]).view(1, N, 1, 1)
        self.register_buffer('yoffset', yoffset)
        self.register_buffer('xoffset', xoffset)
        self.register_buffer('dx', dx)
        self.register_buffer('dy', dy)

    def forward(self, input_map, ref_map, inref_x, inref_y, is_final, iteration_count, input_minWH, ref_minWH):

        b, c, h, w = input_map.size()
        w1 = ref_map.size()[3]
        h1 = ref_map.size()[2]

        for prop_iter in range(iteration_count):
            k = 1
            while k <= input_minWH:  # jumpflooding
                aggregated_x, aggregated_y = self.propagation(inref_x, inref_y, k)
                inref_x, inref_y = self.evaluate(input_map, ref_map,
                                                 aggregated_x, aggregated_y)
                k = k * 2
            k = 1
            while k <= ref_minWH:
                aggregated_x = inref_x.expand(-1, 9, -1, -1)
                aggregated_x = (aggregated_x + k * self.xoffset + (prop_iter % k) * self.dx + w1) % w1
                aggregated_y = inref_y.expand(-1, 9, -1, -1)
                aggregated_y = (aggregated_y + k * self.yoffset + (prop_iter % k) * self.dy + h1) % h1
                inref_x, inref_y = self.evaluate(input_map, ref_map,
                                                 aggregated_x, aggregated_y)
                k = k * 2

        inref_x = inref_x.detach()  # (1,1, h, w)
        inref_y = inref_y.detach()  # (1,1, h, w)

        if is_final:
            max_idx = inref_y * w1 + inref_x
            max_idx = max_idx.view(h, w).long()  # [h,w]
            S = self.evaluate(input_map, ref_map,
                              inref_x, inref_y, is_test=True)  # [B,1,H,W]
            return max_idx, S
        else:
            return inref_x, inref_y


class feature_match_index(nn.Module):
    def __init__(self):
        super(feature_match_index, self).__init__()
        self.PatchMatch = PatchMatch()

    def forward(self, feat_input2,
                feat_ref2,
                prymid_input,
                prymid_ref,
                patch_size=3,
                input_stride=1,
                ref_stride=1,
                is_norm=True,
                norm_input=False):

        feat_input2 = feat_input2.unsqueeze(0)
        feat_ref2 = feat_ref2.unsqueeze(0)
        b, c, h, w = feat_input2.size()
        w1 = feat_ref2.size()[3]
        h1 = feat_ref2.size()[2]
        device = feat_input2.device
        input_map2 = F.unfold(feat_input2, kernel_size=(3, 3)).view(b, -1, h - 2,
                                                                    w - 2)  # [B,C*(N=9),H,W]
        ref_map2 = F.normalize(F.unfold(feat_ref2, kernel_size=(3, 3)), dim=1).view(b, -1, h1 - 2,
                                                                                    w1 - 2)  # [B,C*(N=9),H,W]
        input_minWH = min(input_map2.shape[2] - 1, input_map2.shape[3] - 1)
        ref_minWH = min(ref_map2.shape[2] - 1, ref_map2.shape[3] - 1)
        input_minWH = 2 ** math.floor(math.log2(input_minWH // 8))
        ref_minWH = 2 ** math.floor(math.log2(ref_minWH // 8))
        for i, (feat_input, feat_ref) in enumerate(zip(reversed(prymid_input), reversed(prymid_ref))):
            feat_input = feat_input.unsqueeze(0)
            feat_ref = feat_ref.unsqueeze(0)
            b, c, h, w = feat_input.size()
            w1 = feat_ref.size()[3]
            h1 = feat_ref.size()[2]
            if i == len(prymid_input) - 1:
                input_map = F.unfold(feat_input, kernel_size=(3, 3)).view(b, -1, h - 2,
                                                                          w - 2)  # [B,C*(N=9),H,W]
                ref_map = F.normalize(F.unfold(feat_ref, kernel_size=(3, 3)), dim=1).view(b, -1, h1 - 2,
                                                                                          w1 - 2)  # [B,C*(N=9),H,W]
            else:
                input_map = F.unfold(feat_input, kernel_size=(3, 3), padding=1).view(b, -1, h,
                                                                                     w)  # [B,C*(N=9),H,W]
                ref_map = F.normalize(F.unfold(feat_ref, kernel_size=(3, 3), padding=1), dim=1).view(b, -1, h1,
                                                                                                     w1)  # [B,C*(N=9),H,W]
            b, c, h, w = input_map.size()
            w1 = ref_map.size()[3]
            h1 = ref_map.size()[2]
            if i == 0:
                # random initilization
                wpre1 = w1
                hpre1 = h1
                inref_x = torch.round(torch.rand((b, 1, h, w)) * (w1 - 1)).to(device)  # [b,1,h,w]
                inref_y = torch.round(torch.rand((b, 1, h, w)) * (h1 - 1)).to(device)
                iteration_count = 1

                if i == len(prymid_input) - 1:
                    max_idx, S = self.PatchMatch(input_map, ref_map, inref_x, inref_y, True, iteration_count,
                                                 input_minWH, ref_minWH)
                else:
                    inref_x, inref_y = self.PatchMatch(input_map, ref_map, inref_x, inref_y, False, iteration_count,
                                                       input_minWH, ref_minWH)
            else:
                if i == len(prymid_input) - 1:
                    iteration_count = 4
                    inref_x = torch.round(inref_x * (w1 - 1) / (wpre1 - 1))
                    inref_y = torch.round(inref_y * (h1 - 1) / (hpre1 - 1))
                    inref_x = torch.repeat_interleave(inref_x, 2, 2)
                    inref_x = torch.repeat_interleave(inref_x, 2, 3)
                    inref_y = torch.repeat_interleave(inref_y, 2, 2)
                    inref_y = torch.repeat_interleave(inref_y, 2, 3)
                    inref_x = inref_x[:, :, 1:-1, 1:-1].view(b, 1, h, w)
                    inref_y = inref_y[:, :, 1:-1, 1:-1].view(b, 1, h, w)
                    max_idx, S = self.PatchMatch(input_map, ref_map, inref_x, inref_y, True, iteration_count,
                                                 input_minWH, ref_minWH)
                else:
                    iteration_count = 1
                    inref_x = torch.round(inref_x * (w1 - 1) / (wpre1 - 1))
                    inref_y = torch.round(inref_y * (h1 - 1) / (hpre1 - 1))
                    inref_x = torch.repeat_interleave(inref_x, 2, 2)
                    inref_x = torch.repeat_interleave(inref_x, 2, 3)
                    inref_y = torch.repeat_interleave(inref_y, 2, 2)
                    inref_y = torch.repeat_interleave(inref_y, 2, 3)
                    inref_x, inref_y = self.PatchMatch(input_map, ref_map, inref_x, inref_y, False, iteration_count,
                                                       input_minWH, ref_minWH)
                wpre1 = w1
                hpre1 = h1

        return max_idx, S
