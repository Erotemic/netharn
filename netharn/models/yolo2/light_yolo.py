#
#   Darknet YOLOv2 model
#   Copyright EAVISE
#
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
# import lightnet.data as lnd

__all__ = ['Yolo']


class Yolo(lnn.Darknet):
    """ `Yolo v2`_ implementation with pytorch.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (dict, optional): Dictionary containing `num` and `values` properties with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    .. _Yolo v2: https://github.com/pjreddie/darknet/blob/777b0982322142991e1861161e68e1a01063d76f/cfg/yolo-voc.cfg

    Example:
        >>> from clab.models.yolo2.light_yolo import *
        >>> torch.random.manual_seed(0)
        >>> B, C, Win, Hin = 2, 20, 96, 96
        >>> self = Yolo(num_classes=C, conf_thresh=4.9e-2)
        >>> im_data = torch.randn(B, 3, Hin, Win)
        >>> # the _forward function produces raw YOLO output
        >>> network_output = self.forward(im_data)
        >>> A = self.num_anchors
        >>> Wout, Hout = Win // 32, Hin // 32
        >>> assert list(network_output.shape) == [2, 125, 3, 3]
        >>> assert list(network_output.shape) == [B, A * (C + 5), Wout, Hout]
        >>> # The default postprocess function will construct the bounding boxes
        >>> # Each item in `postout` is a list of detected boxes with columns:
        >>> # x_center, y_center, width, height, confidence, class_id
        >>> postout = self.postprocess(network_output)
        >>> boxes = postout[0].numpy()
        >>> print(boxes)  # xdoc: +IGNORE_WANT
        [[ 0.8342051   0.4984206   5.5057874   3.8463938   0.05158421  5.        ]
         [ 0.84211665  0.16787912  1.5673848   1.9569225   0.05154617 15.        ]
         [ 0.48864678  0.50975     0.8462989   0.79987097  0.04938301 10.        ]]
    """
    def __init__(self, num_classes=20, weights_file=None, conf_thresh=.25,
                 nms_thresh=.4, input_channels=3, anchors=None):
        """ Network initialisation """
        super(Yolo, self).__init__()

        if anchors is None:
            anchors = dict(num=5, values=[1.3221, 1.73145, 3.19275, 4.00944,
                                          5.05587, 8.09892, 9.47112, 4.84053,
                                          11.2364, 10.0071])

        # Parameters
        self.num_classes = num_classes
        self.num_anchors = anchors['num']
        self.anchors = anchors['values']
        self.reduction = 32             # input_dim/output_dim

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchLeaky(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchLeaky(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnn.layer.Conv2dBatchLeaky(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnn.layer.Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchLeaky(512, 64, 1, 1, 0)),
                ('27_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchLeaky((4*64)+1024, 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, self.num_anchors*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        self.load_weights(weights_file)
        # self.loss = lnn.RegionLoss(self)
        from clab.models.yolo2 import light_postproc
        self.postprocess = light_postproc.GetBoundingBoxes(self, conf_thresh, nms_thresh)

    def output_shape_for(self, input_shape):
        b, c, h, w = input_shape
        o = self.num_anchors*(5+self.num_classes)
        return (b, o, h // 32, w // 32)

    def forward(self, x):
        """
        Example:
            >>> from clab.models.yolo2.light_yolo import *
            >>> inp_size = (288, 288)
            >>> self = Yolo(num_classes=20, conf_thresh=0.01, nms_thresh=0.4)
            >>> state_dict = torch.load(demo_weights())['weights']
            >>> self.load_state_dict(state_dict)
            >>> im_data, rgb255 = demo_image(inp_size)
            >>> im_data = torch.cat([im_data, im_data])  # make a batch size of 2
            >>> output = self(im_data)
            >>> # Define remaining params
            >>> orig_sizes = torch.LongTensor([rgb255.shape[0:2][::-1]] * len(im_data))
            >>> postout = self.postprocess(output)
            >>> out_boxes = postout[0][:, 0:4]
            >>> out_scores = postout[0][:, 4]
            >>> out_cxs = postout[0][:, 5]
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> from clab import util
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            >>>  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            >>>  'dog', 'horse', 'motorbike', 'person',
            >>>  'pottedplant', 'sheep', 'sofa', 'train',
            >>>  'tvmonitor')
            >>> import pandas as pd
            >>> cls_names = list(ub.take(label_names, out_cxs.numpy().astype(np.int).tolist()))
            >>> print(pd.DataFrame({'name': cls_names, 'score': out_scores}))
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> #sf = orig_sizes[0].numpy() / (np.array(inp_size) / 32)
            >>> sf = orig_sizes[0].numpy()
            >>> norm_cxywh = util.Boxes(out_boxes.numpy(), 'cxywh')
            >>> xywh = norm_cxywh.asformat('xywh').scale(sf).data
            >>> mplutil.imshow(rgb255, colorspace='rgb')
            >>> mplutil.draw_boxes(xywh)
            >>> mplutil.show_if_requested()
        """
        outputs = []

        outputs.append(self.layers[0](x))
        outputs.append(self.layers[1](outputs[0]))
        # Route : layers=-9
        outputs.append(self.layers[2](outputs[0]))
        # Route : layers=-1,-4
        out = self.layers[3](torch.cat((outputs[2], outputs[1]), 1))

        return out


def demo_weights():
    from os.path import dirname, join
    import lightnet
    dpath = dirname(dirname(lightnet.__file__))
    fpath = join(dpath, 'examples', 'yolo-voc', 'lightnet_weights.pt')
    return fpath


def demo_image(inp_size):
    from clab import util
    import numpy as np
    import cv2
    rgb255 = util.grab_test_image('astro', 'rgb')
    rgb01 = cv2.resize(rgb255, inp_size).astype(np.float32) / 255
    im_data = torch.FloatTensor([rgb01.transpose(2, 0, 1)])
    return im_data, rgb255


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m lightnet.models.network_yolo all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
