import torch.nn.functional as F
from netharn.models.faster_rcnn.utils.config import cfg
from netharn.models.faster_rcnn.fpn.fpn import _FPN
from netharn.models.faster_rcnn.roi_align.modules.roi_align import RoIAlignAvg

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class ResnetFeaturePyramidNetwork(nn.Module):
    """
    This part is only the backbone feature pyramid and nothing else.

    This is a only meant to be used as a modular feature extractor.

    Example:
        >>> self = ResnetFeaturePyramidNetwork()
        >>> print('self = {!r}'.format(self))
    """

    def __init__(self):
        super().__init__()
        self.POOLING_MODE = 'align'
        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # FIXME:
        self.dout_base_model = 256

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_align = RoIAlignAvg(
            self.POOLING_SIZE, self.POOLING_SIZE, 1.0 / 16.0)
        self.grid_size = (cfg.POOLING_SIZE * 2
                          if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE)

        resnet = resnet50()

        self.RCNN_layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.RCNN_layer1 = nn.Sequential(resnet.layer1)
        self.RCNN_layer2 = nn.Sequential(resnet.layer2)
        self.RCNN_layer3 = nn.Sequential(resnet.layer3)
        self.RCNN_layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(
            2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer3 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, im_data):
        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        return rpn_feature_maps

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level.fill_(5)
        if self.POOLING_MODE == 'crop':
            raise NotImplementedError

        elif self.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif self.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        return roi_pool_feat
