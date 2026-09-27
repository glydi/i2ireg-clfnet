"""
I2IReg-ClfNet: cascaded image-to-image regression + ROI-guided classification.

Reference: Öksüz et al., "I2IReg–ClfNet: a cascaded multi-task deep learning framework
for ROI-aware kidney stone detection in abdominal CT images",
Biomed. Signal Process. Control 119 (2026) 109857.

Stage 1  I2IRegNet  DeepLabv3+ (ResNet18 FEB, 16x down-sampling, depthwise ASPP,
                    two up-sampling units) that regresses the kidney ROI image.
Stage 2  ClfNet     dual encoder (trainable shallow CNN + frozen ImageNet CNN),
                    BN -> 1x1 conv -> CBAM -> ReLU -> Dropout -> GAP -> FC -> softmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNorm(nn.Module):
    """Normalises a [0,1] image with ImageNet statistics (buffers move with .to())."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """The '2D Cropping' layer of Fig. 2.2: trims transposed-conv overhang."""
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def dw_pw(cin: int, cout: int, dilation: int = 1) -> nn.Sequential:
    """Grouped (depthwise) 3x3 conv -> pointwise 1x1 conv -> BN -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=dilation, dilation=dilation, groups=cin, bias=False),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------------------
# Stage 1: I2IRegNet (DeepLabv3+ with ResNet18 feature-extraction backbone)
# --------------------------------------------------------------------------------------
class ResNet18FEB(nn.Module):
    """ResNet18 truncated at layer3 -> 16x down-sampling (14x14x256 for 224 input).
    Also returns the 56x56x64 low-level map used by up-sampling unit #1."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3
        self.low_channels, self.high_channels = 64, 256

    def forward(self, x):
        low = self.layer1(self.stem(x))            # 56x56x64
        high = self.layer3(self.layer2(low))       # 14x14x256
        return low, high


class ASPP(nn.Module):
    """Four depthwise-dilated branches (rates 1, 6, 12, 18) + pointwise 256 each -> 1024."""

    def __init__(self, cin: int, cbranch: int = 256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(dw_pw(cin, cbranch, r) for r in rates)
        self.out_channels = cbranch * len(rates)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class I2IRegNet(nn.Module):
    """Image-to-image regression network.

    head="regression": outputs a 3-channel weighting map W; the ROI estimate is
        ROI_hat = W * x_raw  (the 'reference line for weighting' in Fig. 2.1), so pixels
        inside the ROI keep intensities aligned with the original image. Trained with half-MSE.
    head="segmentation": ablation of Sec. 3.4.3.2; outputs 1-channel mask logits,
        ROI_hat = sigmoid(logits) replicated to 3 channels (binary-mask input to ClfNet).
    """

    def __init__(self, pretrained: bool = True, head: str = "regression", img_size: int = 224):
        super().__init__()
        assert head in ("regression", "segmentation")
        self.head, self.img_size = head, img_size
        self.norm = ImageNetNorm()
        self.feb = ResNet18FEB(pretrained)
        self.aspp = ASPP(self.feb.high_channels)
        # Up-sampling unit #1
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(self.aspp.out_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.tconv1 = nn.ConvTranspose2d(256, 256, 8, stride=4)        # 14 -> 60 -> crop 56
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(self.feb.low_channels, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        # Up-sampling unit #2
        self.up2 = nn.Sequential(dw_pw(304, 256), dw_pw(256, 256))
        out_ch = 3 if head == "regression" else 1
        self.mapping = nn.Conv2d(256, out_ch, 1)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, 8, stride=4)  # 56 -> 228 -> crop 224

    def forward(self, x_raw):
        low, high = self.feb(self.norm(x_raw))
        y = self.dec_conv1(self.aspp(high))
        y = center_crop(self.tconv1(y), low.shape[-1])
        y = torch.cat([y, self.dec_conv2(low)], dim=1)                 # 56x56x304
        y = self.mapping(self.up2(y))
        y = center_crop(self.tconv2(y), self.img_size)
        if self.head == "regression":
            return y * x_raw, y            # ROI_hat, raw map
        prob = torch.sigmoid(y)
        return prob.expand(-1, 3, -1, -1), y


# --------------------------------------------------------------------------------------
# Stage 2: ClfNet (dual encoder + CBAM)
# --------------------------------------------------------------------------------------
class ShallowEncoder(nn.Module):
    """Core block = Conv(3x3,Ca)-BN-ReLU-MaxPool-Conv(3x3,Ca)-BN-ReLU (Fig. 2.3).
    Kernel counts from the paper's Bayesian search: 128, 32, 96, 64, 256, (224).
    Blocks that would drop the map below 7x7 are pruned, so a 224 input keeps five
    blocks and yields 7x7x256."""

    def __init__(self, channels=(128, 32, 96, 64, 256), in_ch: int = 3):
        super().__init__()
        layers, c = [], in_ch
        for ca in channels:
            layers += [
                nn.Conv2d(c, ca, 3, padding=1, bias=False), nn.BatchNorm2d(ca), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(ca, ca, 3, padding=1, bias=False), nn.BatchNorm2d(ca), nn.ReLU(inplace=True),
            ]
            c = ca
        self.body = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.body(x)


class DeepEncoder(nn.Module):
    """Frozen ImageNet backbone truncated to a 7x7 feature map."""

    def __init__(self, name: str = "squeezenet", pretrained: bool = True):
        super().__init__()
        self.norm = ImageNetNorm()
        if name == "squeezenet":
            m = tvm.squeezenet1_1(weights=tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None)
            # SqueezeNet's last map is 13x13 for a 224 input; pool it to the shared 7x7 grid.
            self.body = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(7))
            self.out_channels = 512
        elif name == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.body = nn.Sequential(*list(m.children())[:-2])
            self.out_channels = 512
        elif name == "efficientnet_v2_s":
            m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            self.body = m.features
            self.out_channels = 1280
        else:
            raise ValueError(f"unknown deep encoder {name}")
        for p in self.body.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.body.eval()   # frozen: keep BN statistics fixed
        return self

    def forward(self, x):
        return self.body(self.norm(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): channel then spatial."""

    def __init__(self, c: int, reduction: int = 16, k: int = 7):
        super().__init__()
        hidden = max(c // reduction, 1)
        self.mlp = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, c, 1))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class ClfNet(nn.Module):
    """Classification subnetwork. `shallow=False` gives the single-encoder variants of Table 3.4."""

    def __init__(self, deep: str = "squeezenet", shallow: bool = True, cbam: bool = True,
                 num_classes: int = 2, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.deep = DeepEncoder(deep, pretrained)
        self.shallow = ShallowEncoder() if shallow else None
        c = self.deep.out_channels + (self.shallow.out_channels if shallow else 0)
        c_red = round(c / 2)
        self.fuse_bn = nn.BatchNorm2d(c)
        self.reduce = nn.Conv2d(c, c_red, 1)
        self.cbam = CBAM(c_red) if cbam else nn.Identity()
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_red, num_classes)

    def features(self, x):
        feats = [self.deep(x)]
        if self.shallow is not None:
            feats.insert(0, self.shallow(x))
        return self.cbam(self.reduce(self.fuse_bn(torch.cat(feats, 1))))

    def forward(self, x):
        f = self.act(self.features(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.drop(f), 1), 1))


# --------------------------------------------------------------------------------------
# Full cascaded model and baselines
# --------------------------------------------------------------------------------------
class I2IRegClfNet(nn.Module):
    """ROI-aware cascade: x -> I2IRegNet -> ROI_hat -> ClfNet -> logits (end-to-end trainable)."""

    uses_roi = True

    def __init__(self, deep: str = "squeezenet", shallow: bool = True, cbam: bool = True,
                 head: str = "regression", pretrained: bool = True):
        super().__init__()
        self.reg = I2IRegNet(pretrained, head)
        self.clf = ClfNet(deep, shallow, cbam, pretrained=pretrained)

    def forward(self, x):
        roi_hat, raw = self.reg(x)
        return {"logits": self.clf(roi_hat), "roi": roi_hat, "raw": raw}


class FullImageClassifier(nn.Module):
    """Non-ROI baseline: ClfNet (or a frozen single backbone + head) on the whole slice."""

    uses_roi = False

    def __init__(self, deep: str = "squeezenet", shallow: bool = True, cbam: bool = True,
                 pretrained: bool = True):
        super().__init__()
        self.clf = ClfNet(deep, shallow, cbam, pretrained=pretrained)

    def forward(self, x):
        return {"logits": self.clf(x), "roi": None, "raw": None}


def build_model(arch: str, deep: str = "squeezenet", shallow: bool = True, cbam: bool = True,
                head: str = "regression", pretrained: bool = True) -> nn.Module:
    if arch == "i2ireg_clfnet":
        return I2IRegClfNet(deep, shallow, cbam, head, pretrained)
    if arch == "full_image":
        return FullImageClassifier(deep, shallow, cbam, pretrained)
    raise ValueError(arch)


def count_params(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable
