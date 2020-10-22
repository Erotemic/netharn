"""
Experiment Script Related to Pytorch Memory Leak Issue

References:
    https://github.com/pytorch/pytorch/issues/13246
    https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import psutil
import ubelt as ub
import sys


class DataIter(Dataset):
    def __init__(self, storage_mode='numpy', return_mode='tensor', total=24e7):
        self.return_mode = return_mode
        self.storage_mode = storage_mode

        assert self.return_mode in {'tensor', 'dict', 'tuple', 'list'}

        if storage_mode == 'numpy':
            self.data = np.array([x for x in range(int(total))])
        elif storage_mode == 'python':
            self.data = [x for x in range(int(total))]
        elif storage_mode == 'ndsampler':
            import ndsampler
            assert total <= 1000
            self.data = ndsampler.CocoSampler.demo('shapes{}'.format(total))
        else:
            raise KeyError(storage_mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.storage_mode == 'ndsampler':
            data = self.data.load_item(idx)['im'].ravel()[0:1].astype(np.float32)
            data_pt = torch.from_numpy(data)
        else:
            data = self.data[idx]
            data = np.array([data], dtype=np.int64)
            data_pt = torch.tensor(data)

        if self.return_mode == 'tensor':
            item = data_pt
        elif self.return_mode == 'dict':
            item = {
                'data': data_pt
            }
        elif self.return_mode == 'tuple':
            item = (data_pt,)
        elif self.return_mode == 'list':
            item = [data_pt]
        return item


def getsize(*objs):
    """
    sum size of object & members.
    https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    """
    import sys
    from types import ModuleType, FunctionType
    from gc import get_referents
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    blocklist = (type, ModuleType, FunctionType)
    # if isinstance(obj, blocklist):
    #     raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = objs
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, blocklist) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size, len(seen_ids)


def byte_str(num, unit='auto', precision=2):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Args:
        num (int): number of bytes
        unit (str): which unit to use, can be auto, B, KB, MB, GB, TB, PB, EB,
            ZB, or YB.

    References:
        https://en.wikipedia.org/wiki/Orders_of_magnitude_(data)

    Returns:
        str: string representing the number of bytes with appropriate units

    Example:
        >>> num_list = [1, 100, 1024,  1048576, 1073741824, 1099511627776]
        >>> result = ub.repr2(list(map(byte_str, num_list)), nl=0)
        >>> print(result)
        ['0.00 KB', '0.10 KB', '1.00 KB', '1.00 MB', '1.00 GB', '1.00 TB']
    """
    abs_num = abs(num)
    if unit == 'auto':
        if abs_num < 2.0 ** 10:
            unit = 'KB'
        elif abs_num < 2.0 ** 20:
            unit = 'KB'
        elif abs_num < 2.0 ** 30:
            unit = 'MB'
        elif abs_num < 2.0 ** 40:
            unit = 'GB'
        elif abs_num < 2.0 ** 50:
            unit = 'TB'
        elif abs_num < 2.0 ** 60:
            unit = 'PB'
        elif abs_num < 2.0 ** 70:
            unit = 'EB'
        elif abs_num < 2.0 ** 80:
            unit = 'ZB'
        else:
            unit = 'YB'
    if unit.lower().startswith('b'):
        num_unit = num
    elif unit.lower().startswith('k'):
        num_unit =  num / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        num_unit =  num / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        num_unit = num / (2.0 ** 30)
    elif unit.lower().startswith('t'):
        num_unit = num / (2.0 ** 40)
    elif unit.lower().startswith('p'):
        num_unit = num / (2.0 ** 50)
    elif unit.lower().startswith('e'):
        num_unit = num / (2.0 ** 60)
    elif unit.lower().startswith('z'):
        num_unit = num / (2.0 ** 70)
    elif unit.lower().startswith('y'):
        num_unit = num / (2.0 ** 80)
    else:
        raise ValueError('unknown num={!r} unit={!r}'.format(num, unit))
    return ub.repr2(num_unit, precision=precision) + ' ' + unit


def main(storage_mode='numpy', return_mode='tensor', total=24e5, shuffle=True):
    """
    Args:
        storage_mode : how the dataset is stored in backend datasets

        return_mode : how each data item is returned

        total : size of backend storage

    """
    mem = psutil.virtual_memory()
    start_mem = mem.used
    mem_str = byte_str(start_mem)
    print('Starting used system memory = {!r}'.format(mem_str))

    train_data = DataIter(
        storage_mode=storage_mode,
        return_mode=return_mode,
        total=total)
    # self = train_data

    if storage_mode == 'numpy':
        total_storate_bytes = train_data.data.dtype.itemsize * train_data.data.size
    else:
        total_storate_bytes = sys.getsizeof(train_data.data)
        # total_storate_bytes = getsize(self.data)
    print('total_storage_size = {!r}'.format(byte_str(total_storate_bytes)))

    mem = psutil.virtual_memory()
    mem_str = byte_str(mem.used - start_mem)
    print('After init DataIter   memory = {!r}'.format(mem_str))

    print('shuffle = {!r}'.format(shuffle))

    num_workers = 2
    train_loader = DataLoader(train_data, batch_size=300,
                              shuffle=shuffle,
                              drop_last=True,
                              pin_memory=False,
                              num_workers=num_workers)

    used_nbytes = psutil.virtual_memory().used - start_mem
    print('After init DataLoader memory = {!r}'.format(byte_str(used_nbytes)))

    if True:
        # Estimate peak usage
        import gc
        all_obj_nbytes, num_objects = getsize(*gc.get_objects())
        python_ptr_size = int((np.log2(sys.maxsize) + 1) / 8)
        assert python_ptr_size == 8, 'should be 8 bytes on 64bit python'
        all_ptr_nbytes = (num_objects * python_ptr_size)

        prog_nbytes_estimated_1 = all_ptr_nbytes + all_obj_nbytes
        prog_nbytes_measured_2 = psutil.virtual_memory().used - start_mem
        print('prog_nbytes_estimated_1 = {!r}'.format(byte_str(prog_nbytes_estimated_1)))
        print('prog_nbytes_measured_2  = {!r}'.format(byte_str(prog_nbytes_measured_2)))

        peak_bytes_est1 = prog_nbytes_estimated_1 * (num_workers + 1)
        peak_bytes_est2 = prog_nbytes_measured_2 * (num_workers + 1)
        print('peak_bytes_est1 = {!r}'.format(byte_str(peak_bytes_est1)))
        print('peak_bytes_est2 = {!r}'.format(byte_str(peak_bytes_est2)))

    max_bytes = -float('inf')
    prog = ub.ProgIter(train_loader)
    for item in prog:
        used_bytes = psutil.virtual_memory().used - start_mem
        max_bytes = max(max_bytes, used_bytes)
        prog.set_extra(' Mem=' + byte_str(used_bytes))

    used_bytes = psutil.virtual_memory().used - start_mem
    print('measured final usage: {}'.format(byte_str(used_bytes)))
    print('measured peak usage:  {}'.format(byte_str(max_bytes)))


if __name__ == '__main__':
    """
    CommandLine:
        python debug_memory.py numpy tensor --total=24e5 --shuffle=True

        cd ~/code/netharn/dev

        python debug_memory.py --storage_mode=numpy --total=24e5 --shuffle=True
        python debug_memory.py --storage_mode=numpy --total=24e5 --shuffle=False
        python debug_memory.py --storage_mode=python --total=24e5 --shuffle=True
        python debug_memory.py --storage_mode=python --total=24e5 --shuffle=False

        python debug_memory.py --storage_mode=ndsampler --total=1000 --shuffle=True

        python debug_memory.py numpy dict 24e5
        python debug_memory.py python list 24e7

    Conclusions:

        * It seems like it is ok if the return type is a dictionary
          the problem seems to be localized to the storage type.
    """
    import fire
    fire.Fire(main)

"""

@VitalyFedyunin Let me see if I understand correctly, when you access an item
in a list you create a new reference to it, which will force its refcount to be
incremented (i.e. be written to).

pages are typically 4096 bytes.

"""
