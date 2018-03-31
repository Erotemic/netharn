import torch.utils.data as torch_data
import torch
import ubelt as ub
import numpy as np  # NOQA


default_collate = torch_data.dataloader.default_collate


def list_collate(inbatch):
    """
    Used for detection datasets with boxes.

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     boxes = torch.rand(rng.randint(0, 4), 4)
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = list_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert len(out_batch[1][0]) == bsize

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     boxes = torch.empty(0, 4)
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = list_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert len(out_batch[1][0]) == bsize
    """
    try:
        # if True:
        if torch.is_tensor(inbatch[0]):
            num_items = [len(item) for item in inbatch]
            if ub.allsame(num_items):
                if len(num_items) == 0 or num_items[0] == 0:
                    batch = inbatch
                else:
                    batch = default_collate(inbatch)
            else:
                batch = inbatch
        else:
            batch = [list_collate(item) for item in list(map(list, zip(*inbatch)))]
    except Exception as ex:
        print('Failed to collate inbatch={}'.format(inbatch))
        import utool
        utool.embed()
        raise
    return batch
    # else:
    #     # we know the order of data in __getitem__ so we can choose not to
    #     # stack the variable length bboxes and labels
    #     inbatchT = list(map(list, zip(*inbatch)))
    #     inimgs, inlabels = inbatchT
    #     imgs = default_collate(inimgs)

    #     # Just transpose the list if we cant collate the labels
    #     # However, try to collage each part.
    #     n_labels = len(inlabels[0])
    #     labels = [None] * n_labels
    #     for i in range(n_labels):
    #         simple = [x[i] for x in inlabels]
    #         if ub.allsame(map(len, simple)):
    #             labels[i] = default_collate(simple)
    #         else:
    #             labels[i] = simple

    #     batch = imgs, labels
    #     return batch


def padded_collate(inbatch, fill_value=-1):
    """
    Used for detection datasets with boxes.

    Example:
        >>> from clab.data.collate import *
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 7
        >>> for i in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     n = 11 if i == 3 else rng.randint(0, 11)
        >>>     boxes = torch.rand(n, 4)
        >>>     item = (img, boxes)
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert list(out_batch[1].shape) == [bsize, 11, 4]

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     boxes = torch.empty(0, 4)
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> #assert list(out_batch[1][0].shape) == [bsize, 0, 4]
        >>> assert list(out_batch[1][0].shape) == [0]

    Example:
        >>> inbatch = [torch.rand(4, 4), torch.rand(8, 4),
        >>>            torch.rand(0, 4), torch.rand(3, 4),
        >>>            torch.rand(0, 4), torch.rand(1, 4)]
        >>> out_batch = padded_collate(inbatch)
        >>> assert list(out_batch.shape) == [6, 8, 4]
    """
    try:
        if torch.is_tensor(inbatch[0]):
            num_items = [len(item) for item in inbatch]
            if ub.allsame(num_items):
                if len(num_items) == 0:
                    batch = torch.empty(0)
                elif num_items[0] == 0:
                    batch = torch.empty(0)
                    # batch = torch.Tensor(inbatch)
                else:
                    batch = default_collate(inbatch)
            else:
                max_size = max(num_items)
                real_tail_shape = None
                for item in inbatch:
                    if item.numel():
                        tail_shape = item.shape[1:]
                        if real_tail_shape is not None:
                            assert real_tail_shape == tail_shape
                        real_tail_shape = tail_shape

                padded_inbatch = []
                for item in inbatch:
                    n_extra = max_size - len(item)
                    if n_extra > 0:
                        shape = (n_extra,) + tuple(real_tail_shape)
                        extra = torch.full(shape, fill_value=fill_value,
                                           dtype=item.dtype)
                        padded_item = torch.cat([item, extra], dim=0)
                        padded_inbatch.append(padded_item)
                    else:
                        padded_inbatch.append(item)
                batch = inbatch
                batch = default_collate(padded_inbatch)
        else:
            batch = [padded_collate(item) for item in list(map(list, zip(*inbatch)))]
    except Exception as ex:
        print('Failed to collate inbatch={}'.format(inbatch))
        import utool
        utool.embed()
        raise
    return batch


if __name__ == '__main__':
    """
    CommandLine:
        python -m clab.data.collate all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
