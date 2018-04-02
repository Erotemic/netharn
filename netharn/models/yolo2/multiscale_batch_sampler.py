import torch.utils.data.sampler as torch_sampler
import torch


class MultiScaleBatchSampler(torch_sampler.BatchSampler):
    """
    Indicies returned in the batch are tuples indicating data index and scale
    index. Requires that dataset has a `multi_scale_inp_size` attribute.

    Example:
        >>> import torch.utils.data as torch_data
        >>> class DummyDatset(torch_data.Dataset):
        >>>     def __init__(self):
        >>>         super(DummyDatset, self).__init__()
        >>>         self.multi_scale_inp_size = [1, 2, 3, 4]
        >>>     def __len__(self):
        >>>         return 34
        >>> batch_size = 16
        >>> data_source = DummyDatset()
        >>> rand = MultiScaleBatchSampler(data_source, shuffle=1)
        >>> seq = MultiScaleBatchSampler(data_source, shuffle=0)
        >>> rand_idxs = list(iter(rand))
        >>> seq_idxs = list(iter(seq))
        >>> assert len(rand_idxs[0]) == 16
        >>> assert len(rand_idxs[0][0]) == 2
        >>> assert len(rand_idxs[-1]) == 2
        >>> assert {len({x[1] for x in xs}) for xs in rand_idxs} == {1}
        >>> assert {x[1] for xs in seq_idxs for x in xs} == {0}
    """

    def __init__(self, data_source, shuffle=False, batch_size=16,
                 drop_last=False, resample_frequency=10):
        if shuffle:
            self.sampler = torch_sampler.RandomSampler(data_source)
        else:
            self.sampler = torch_sampler.SequentialSampler(data_source)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_scales = len(data_source.multi_scale_inp_size)
        self.resample_frequency = resample_frequency

    def __iter__(self):
        batch = []
        if self.shuffle:
            scale_index = int(torch.rand(1) * self.num_scales)
        else:
            scale_index = 0

        for idx in self.sampler:
            batch.append((int(idx), scale_index))
            if len(batch) == self.batch_size:
                yield batch
                if self.shuffle and idx % self.resample_frequency == 0:
                    # choose a new scale index every 10 batches
                    scale_index = int(torch.rand(1) * self.num_scales)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.models.yolo2.multiscale_batch_sampler all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
