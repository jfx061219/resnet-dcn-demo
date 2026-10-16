import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# ResNet blocks: Basic/Bottleneck
# (API compatible with torchvision.models.resnet blocks)
# -------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both convs are 3x3
        self.conv1 = conv3x3(inplanes, planes, stride, groups, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # 1x1 conv
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 3x3 conv
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 1x1 conv
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# -------------------------
# Deformable conv wrapper
# - supports:
#    (1) torchvision.ops.DeformConv2d (no modulation)
#    (2) mmcv.ops.DeformConv2d / ModulatedDeformConv2d (if modulation=True)
# - If neither present, raises ImportError with install hints.
# -------------------------
class DeformConv2d(nn.Module):
    """
    Deformable convolution 2D wrapper.
    - If modulation=False: tries torchvision.ops.DeformConv2d or mmcv.ops.DeformConv2d.
    - If modulation=True: tries mmcv.ops.ModulatedDeformConv2d (DCNv2).
    Note: offset (and mask) are predicted from input via a small conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, modulation=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.modulation = modulation

        # try to import available optimized ops
        self._tv_deform = None
        self._mmcv_deform = None
        self._mmcv_modulated = None
        try:
            from torchvision.ops import DeformConv2d as TVDeformConv2d
            self._tv_deform = TVDeformConv2d
        except Exception:
            self._tv_deform = None

        try:
            # mmcv may provide DeformConv2d and ModulatedDeformConv2d
            from mmcv.ops import DeformConv2d as MMCVDeformConv2d
            from mmcv.ops import ModulatedDeformConv2d as MMCVModulatedDeformConv2d
            self._mmcv_deform = MMCVDeformConv2d
            self._mmcv_modulated = MMCVModulatedDeformConv2d
        except Exception:
            # mmcv import failed or class not present
            # try to import only DeformConv2d if available
            try:
                from mmcv.ops import DeformConv2d as MMCVDeformConv2d
                self._mmcv_deform = MMCVDeformConv2d
            except Exception:
                self._mmcv_deform = None
            self._mmcv_modulated = None

        # decide backend
        if self.modulation and self._mmcv_modulated is None:
            raise ImportError(
                "Modulated deform conv (DCNv2) requested but mmcv.ops.ModulatedDeformConv2d not found. "
                "Install mmcv-full that matches your CUDA / PyTorch: https://github.com/open-mmlab/mmcv"
            )

        if not self.modulation and self._tv_deform is None and self._mmcv_deform is None:
            raise ImportError(
                "No deform conv backend found. Install torchvision>=0.9 (has torchvision.ops.DeformConv2d) "
                "or mmcv-full. Example: pip install torchvision --upgrade  OR follow mmcv install guide."
            )

        # offset (and mask) prediction conv
        if self.modulation:
            offset_channels = (2 + 1) * kernel_size * kernel_size
        else:
            offset_channels = 2 * kernel_size * kernel_size

        # we predict offsets/masks from input; stride/padding/dilation same as deform conv usage
        self.conv_offset = nn.Conv2d(in_channels, offset_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, bias=True)
        # init to zero so initial offset = 0 (behaves like regular conv)
        nn.init.constant_(self.conv_offset.weight, 0.)
        if self.conv_offset.bias is not None:
            nn.init.constant_(self.conv_offset.bias, 0.)

        # instantiate actual deform conv module for forward call
        if self.modulation:
            # mmcv modulated deform conv expects (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups)
            self.deform_conv = self._mmcv_modulated(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=bias
            )
        else:
            if self._tv_deform is not None:
                # torchvision.ops.DeformConv2d signature: DeformConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False)
                self.deform_conv = self._tv_deform(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, bias=bias
                )
            else:
                # fallback to mmcv DeformConv2d
                self.deform_conv = self._mmcv_deform(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, bias=bias
                )

    def forward(self, x):
        # predict offsets (and mask if modulated)
        out = self.conv_offset(x)
        if self.modulation:
            # split into offset and mask
            o1, o2, mask = torch.split(out, [self.kernel_size*self.kernel_size, self.kernel_size*self.kernel_size, self.kernel_size*self.kernel_size], dim=1)
            offset = torch.cat([o1, o2], dim=1)
            mask = torch.sigmoid(mask)
            # mmcv ModulatedDeformConv2d expects (x, offset, mask)
            return self.deform_conv(x, offset, mask)
        else:
            offset = out
            # torchvision DeformConv2d module expects (x, offset)
            # mmcv.DeformConv2d expects (x, offset) as well
            return self.deform_conv(x, offset)

if __name__ == "__main__":
    # 模拟输入：batch=2, channel=64, H=32, W=32
    dummy = torch.randn(2, 64, 32, 32)

    print("=== Test BasicBlock ===")
    bb = BasicBlock(inplanes=64, planes=64)
    y_bb = bb(dummy)
    print(f"Input: {dummy.shape}, Output: {y_bb.shape}")

    print("\n=== Test Bottleneck ===")
    downsample = nn.Sequential(
    conv1x1(64, 64 * Bottleneck.expansion),
    nn.BatchNorm2d(64 * Bottleneck.expansion),
)
    btl = Bottleneck(inplanes=64, planes=64, downsample=downsample)
    y_btl = btl(dummy)
    print(f"Input: {dummy.shape}, Output: {y_btl.shape}")

    print("\n=== Test Deformable Conv v1 (torchvision backend) ===")
    dcn = DeformConv2d(in_channels=64, out_channels=64, kernel_size=3, modulation=False)
    y_dcn = dcn(dummy)
    print(f"Input: {dummy.shape}, Output: {y_dcn.shape}")

    # DCNv2 测试（需要mmcv，可选，注释掉也行）
    # print("\n=== Test Deformable Conv v2 (modulated, mmcv required) ===")
    # dcn_v2 = DeformConv2d(in_channels=64, out_channels=64, kernel_size=3, modulation=True)
    # y_dcn_v2 = dcn_v2(dummy)
    # print(f"Input: {dummy.shape}, Output: {y_dcn_v2.shape}")