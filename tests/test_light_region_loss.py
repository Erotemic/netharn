import torch
# from netharn.models.yolo2.light_region_loss import RegionLoss
# from netharn.models.yolo2.light_yolo import Yolo
# import numpy as np


def as_anno(class_id, x_center, y_center, w, h, Win, Hin):
    """
    Construct an BramBox annotation using the basic YOLO box format
    """
    from brambox.boxes.annotations import Annotation
    anno = Annotation()
    anno.class_id = class_id
    anno.x_top_left = (x_center - w / 2) * Win
    anno.y_top_left = (y_center - h / 2) * Hin
    anno.width, anno.height = w * Win, h * Hin
    return anno


def demodata_targets(input_size):
    """
    Construct the same boxes in Brambox and Tensor format

    Args:
        input_size (tuple): width, height of input to the network
    """
    Win, Hin = input_size
    target_brambox = [
        # boxes for batch item 1
        [as_anno(0, 0.50, 0.50, 1.00, 1.00, Win, Hin),
         as_anno(1, 0.32, 0.42, 0.22, 0.12, Win, Hin)],
        # boxes for batch item 2 (it has no objects!)
        []
    ]

    target_tensor = torch.FloatTensor([
        # boxes for batch item 1
        [[0, 0.50, 0.50, 1.00, 1.00],
         [1, 0.32, 0.42, 0.22, 0.12]],
        # boxes for batch item 2 (it has no objects, note the pad!)
        [[-1, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0]],
    ])
    return target_brambox, target_tensor


# def test_tensor_brambox_compat_ground_truth_encoding():
#     """
#     Check that brambox and tensor boxes produce the same ground truth encoding
#     """

#     network = Yolo(num_classes=2, conf_thresh=4e-2)
#     self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)

#     Win, Hin = 96, 96
#     target_brambox, target_tensor = demodata_targets(input_size=(Win, Hin))

#     # convert network input size (e.g. 416x416) to output size (e.g. 13x13)
#     nW, nH = (Win // self.reduction, Hin // self.reduction)

#     pred_boxes = torch.rand(90, 4)

#     _ret_ten = self._build_targets_tensor(pred_boxes, target_tensor, nH, nW)
#     # _ret_box = self._build_targets_brambox(pred_boxes, target_brambox, nH, nW)

#     keys = ['coord_mask', 'conf_mask', 'cls_mask', 'tcoord', 'tconf', 'tcls']

#     errors = []
#     for key, item1, item2 in zip(keys, _ret_box, _ret_ten):
#         bad_flags = ~np.isclose(item1, item2)
#         if np.any(bad_flags):
#             bad_idxs = np.where(bad_flags)
#             msg = 'key={} did not agree at indices {}. values1={} values2={}'.format(
#                 key, bad_idxs, item1[bad_flags], item2[bad_flags])
#             errors.append(msg)

#     if errors:
#         raise AssertionError('\n---\n'.join(errors))


# def test_tensor_brambox_compat_ground_truth_loss():
#     """
#     Check that brambox and tensor boxes produce the same ground truth encoding
#     """
#     network = Yolo(num_classes=2, conf_thresh=4e-2)
#     self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)

#     # Win, Hin = 96, 96
#     Win, Hin = 416, 416
#     target_brambox, target_tensor = demodata_targets(input_size=(Win, Hin))

#     # construct dummy data and run it through the network
#     im_data = torch.randn(len(target_brambox), 3, Hin, Win)
#     output = network.forward(im_data)
#     print('')

#     # Brambox and Tensor formats should produce the same loss
#     loss_brambox = self(output, target_brambox).data.cpu().numpy()
#     loss_tensor = self(output, target_tensor).data.cpu().numpy()
#     print('loss_brambox = {!r}'.format(loss_brambox))
#     print('loss_tensor = {!r}'.format(loss_tensor))
#     assert np.all(np.isclose(loss_brambox, loss_tensor))

#     # Brambox and Tensor formats should produce the same loss
#     loss_brambox = self(output, target_brambox, seen=9999).data.cpu().numpy()
#     loss_tensor = self(output, target_tensor, seen=9999).data.cpu().numpy()
#     print('loss_brambox = {!r}'.format(loss_brambox))
#     print('loss_tensor = {!r}'.format(loss_tensor))
#     assert np.all(np.isclose(loss_brambox, loss_tensor))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/tests/test_light_region_loss.py all
        pytest ~/code/netharn/tests/test_light_region_loss.py -s
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
