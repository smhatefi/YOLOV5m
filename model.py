# https://www.researchgate.net/figure/YOLOv5-architecture-The-YOLO-network-consists-of-three-main-parts-Backbone-Neck-and_fig5_355962110
import time
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import config
from utils.utils import check_size, count_parameters


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        """
        Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        `(b, c*s*s, h//s, w//s)`.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        """
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor x to expand spatial dimensions by redistributing channels, requiring C / gain^2 ==
        0.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)



# # performs a convolution, a batch_norm and then applies a SiLU activation function
# class CBL(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(CBL, self).__init__()

#         conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

#         self.cbl = nn.Sequential(
#             conv,
#             bn,
#             # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         #print(self.cbl(x).shape)
#         return self.cbl(x)


# # which is just a residual block
# class Bottleneck(nn.Module):
#     """
#     Parameters:
#         in_channels (int): number of channel of the input tensor
#         out_channels (int): number of channel of the output tensor
#         width_multiple (float): it controls the number of channels (and weights)
#                                 of all the convolutions beside the
#                                 first and last one. If closer to 0,
#                                 the simpler the modelIf closer to 1,
#                                 the model becomes more complex
#     """
#     def __init__(self, in_channels, out_channels, width_multiple=1):
#         super(Bottleneck, self).__init__()
#         c_ = int(width_multiple*in_channels)
#         self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
#         self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         return self.c2(self.c1(x)) + x


# # kind of CSP backbone (https://arxiv.org/pdf/1911.11929v1.pdf)
# class C3(nn.Module):
#     """
#     Parameters:
#         in_channels (int): number of channel of the input tensor
#         out_channels (int): number of channel of the output tensor
#         width_multiple (float): it controls the number of channels (and weights)
#                                 of all the convolutions beside the
#                                 first and last one. If closer to 0,
#                                 the simpler the modelIf closer to 1,
#                                 the model becomes more complex
#         depth (int): it controls the number of times the bottleneck (residual block)
#                         is repeated within the C3 block
#         backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
#                             it will be composed by bottlenecks 2 (check in the image linked below)
#         https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png

#     """
#     def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
#         super(C3, self).__init__()
#         c_ = int(width_multiple*in_channels)

#         self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
#         self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
#         if backbone:
#             self.seq = nn.Sequential(
#                 *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
#             )
#         else:
#             self.seq = nn.Sequential(
#                 *[nn.Sequential(
#                     CBL(c_, c_, 1, 1, 0),
#                     CBL(c_, c_, 3, 1, 1)
#                 ) for _ in range(depth)]
#             )
#         self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
#         return self.c_out(x)


# # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
# class SPPF(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SPPF, self).__init__()

#         c_ = int(in_channels//2)

#         self.c1 = CBL(in_channels, c_, 1, 1, 0)
#         self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
#         self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

#     def forward(self, x):
#         x = self.c1(x)
#         pool1 = self.pool(x)
#         pool2 = self.pool(pool1)
#         pool3 = self.pool(pool2)

#         return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))


# # in the PANET the C3 block is different: no more CSP but a residual block composed
# # a sequential branch of n SiLUs and a skipped branch with one SiLU
# class C3_NECK(nn.Module):
#     def __init__(self, in_channels, out_channels, width, depth):
#         super(C3_NECK, self).__init__()
#         c_ = int(in_channels*width)
#         self.in_channels = in_channels
#         self.c_ = c_
#         self.out_channels = out_channels
#         self.c_skipped = CBL(in_channels, c_, 1, 1, 0)
#         self.c_out = CBL(c_*2, out_channels, 1, 1, 0)
#         self.silu_block = self.make_silu_block(depth)

#     def make_silu_block(self, depth):
#         layers = []
#         for i in range(depth):
#             if i == 0:
#                 layers.append(CBL(self.in_channels, self.c_, 1, 1, 0))
#             elif i % 2 == 0:
#                 layers.append(CBL(self.c_, self.c_, 3, 1, 1))
#             elif i % 2 != 0:
#                 layers.append(CBL(self.c_, self.c_, 1, 1, 0))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         return self.c_out(torch.cat([self.silu_block(x), self.c_skipped(x)], dim=1))


# class HEADS(nn.Module):
#     def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
#         super(HEADS, self).__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(anchors)  # number of detection layers
#         self.naxs = len(anchors[0])

#         # https://pytorch.org/docs/stable/generated/torch.nn.Module.html command+f register_buffer
#         # has the same result as self.anchors = anchors but, it's a way to register a buffer (make
#         # a variable available in runtime) that should not be considered a model parameter
#         self.stride = [8, 16, 32]

#         # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
#         anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
#         self.register_buffer('anchors', anchors_)  # shape(nl,na,2)

#         self.out_convs = nn.ModuleList()
#         for in_channels in ch:
#             self.out_convs += [
#                 nn.Conv2d(in_channels=in_channels, out_channels=(5+self.nc) * self.naxs, kernel_size=1)
#             ]

#     def forward(self, x):
#         for i in range(self.nl):
#             # performs out_convolution and stores the result in place
#             x[i] = self.out_convs[i](x[i])

#             bs, _, grid_y, grid_x = x[i].shape
#             # reshaping output to be (bs, n_scale_predictions, n_grid_y, n_grid_x, 5 + num_classes)
#             # why .permute? Here https://github.com/ultralytics/yolov5/issues/10524#issuecomment-1356822063
#             x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

#         return x


class YOLOV5m(nn.Module):
    def __init__(self, first_out, nc=80, anchors=(),
                 ch=(), inference=False):
        super(YOLOV5m, self).__init__()
        self.inference = inference

        self.Detect = nn.ModuleList()
        self.Detect += [
            nn.Conv2d(in_channels=first_out*4, out_channels=255, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=first_out*8, out_channels=255, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=first_out*16, out_channels=255, kernel_size=1, stride=1),
        ]

        self.model = nn.Sequential(
            Conv(c1=3, c2=first_out, k=6, s=2, p=2),
            Conv(c1=first_out, c2=first_out*2, k=3, s=2, p=1),
            C3(c1=first_out*2, c2=first_out*2, e=0.5, n=2),
            Conv(c1=first_out*2, c2=first_out*4, k=3, s=2, p=1),
            C3(c1=first_out*4, c2=first_out*4, e=0.5, n=4),
            Conv(c1=first_out*4, c2=first_out*8, k=3, s=2, p=1),
            C3(c1=first_out*8, c2=first_out*8, e=0.5, n=6),
            Conv(c1=first_out*8, c2=first_out*16, k=3, s=2, p=1),
            C3(c1=first_out*16, c2=first_out*16, e=0.5, n=2),
            SPPF(c1=first_out*16, c2=first_out*16),
            Conv(c1=first_out*16, c2=first_out*8, k=1, s=1),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            Concat(),
            C3(c1=first_out*16, c2=first_out*8, e=0.5, n=2),
            Conv(c1=first_out*8, c2=first_out*4, k=1, s=1),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            Concat(),
            C3(c1=first_out*8, c2=first_out*4, e=0.5, n=2),
            Conv(c1=first_out*4, c2=first_out*4, k=3, s=2, p=1),
            Concat(),
            C3(c1=first_out*8, c2=first_out*8, e=0.5, n=2),
            Conv(c1=first_out*8, c2=first_out*8, k=3, s=2, p=1),
            Concat(),
            C3(c1=first_out*16, c2=first_out*16, e=0.5, n=2),
            self.Detect
        )

    def forward(self, x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        return self.model(x)
        # assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        # backbone_connection = []
        # neck_connection = []
        # outputs = []
        # for idx, layer in enumerate(self.backbone):
        #     # takes the out of the 2nd and 3rd C3 block and stores it
        #     x = layer(x)
        #     if idx in [4, 6]:
        #         backbone_connection.append(x)

        # for idx, layer in enumerate(self.neck):
        #     if idx in [0, 2]:
        #         x = layer(x)
        #         neck_connection.append(x)
        #         x = Resize([x.shape[2]*2, x.shape[3]*2], interpolation=InterpolationMode.NEAREST)(x)
        #         x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

        #     elif idx in [4, 6]:
        #         x = layer(x)
        #         x = torch.cat([x, neck_connection.pop(-1)], dim=1)

        #     elif (isinstance(layer, C3_NECK) and idx > 2) or (isinstance(layer, C3) and idx > 2):
        #         x = layer(x)
        #         outputs.append(x)

        #     else:
        #         x = layer(x)
                
        # return self.head(outputs)


if __name__ == "__main__":
    batch_size = 2
    image_height = 640
    image_width = 640
    nc = 80
    anchors = config.ANCHORS
    x = torch.rand(batch_size, 3, image_height, image_width)
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)
    
    print(model)

    start = time.time()
    out = model(x)
    end = time.time()

    assert out[0].shape == (batch_size, 3, image_height//8, image_width//8, nc + 5)
    assert out[1].shape == (batch_size, 3, image_height//16, image_width//16, nc + 5)
    assert out[2].shape == (batch_size, 3, image_height//32, image_width//32, nc + 5)

    print("Success!")
    print("feedforward took {:.2f} seconds".format(end - start))

    """count_parameters(model)
    check_size(model)
    model.half()
    check_size(model)"""


