"""
The X Processing Unit --- an agnostic torch device.

An XPU is an abstracted (X) procesesing unit (PU) with a common API for running
torch operations on a CPU, GPU, or many GPUs.
"""
from __future__ import absolute_import, division, print_function
import ubelt as ub
import warnings
import torch
import six
import os
from netharn import util
from torch._six import container_abcs


__all__ = ['XPU']

# try:
# minimum memory (MB) needed for auto to resolve to GPU by default
NETHARN_MIN_MB = int(os.environ.get('NETHARN_MIN_MB', 6000))
# except Exception:
#     NETHARN_MIN_MB = 6000


# if torch.__version__.startswith('0.3'):
#     _TENSOR_TYPES = (torch._TensorBase, torch.autograd.Variable)
# else:
_TENSOR_TYPES = (torch.Tensor, torch.autograd.Variable)


class MountedModel(torch.nn.Module, util.ModuleMixin):
    """
    Abstraction of DataParallel and DataSerial
    """

    def receptive_field_for(self, input_field=None):
        import netharn as nh
        return nh.ReceptiveFieldFor(self.module)(input_field)


class DataParallel(torch.nn.DataParallel, MountedModel):
    """
    Hack to redefine DataParallel such that it shares a base with DataSerial
    """
    pass


class DataSerial(MountedModel):
    """
    Wraper to create consistent API with DataParallel
    """
    def __init__(self, module):
        super(DataSerial, self).__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)


class XPU(ub.NiceRepr):
    """
    A processing device or devices: either a CPU, GPU, or multiple GPUS.

    Args:
        item (None, int, or list): None for cpu, an int for a gpu, or a list of
            ints for multiple gpus.
    TODO:
        distributed processing

    CommandLine:
        python -m netharn.device XPU

    Example:
        >>> print(str(XPU(None)))
        CPU
        >>> print(str(XPU(0, check=False)))
        GPU(0)
        >>> print(str(XPU([1, 2, 3], check=False)))
        GPU(1*,2,3)
        >>> import pytest
        >>> with pytest.raises(IndexError):
        >>>     print(str(XPU([], check=False)))
    """
    def __init__(xpu, item=None, check=True):
        xpu._main_device_id = None
        xpu._device_ids = None
        xpu.mode = None

        # For context manager
        xpu._cuda_device = None

        if isinstance(item, torch.device):
            if item.type == 'cpu':
                item = None
            elif item.type == 'cuda':
                item = item.index or 0
            else:
                raise KeyError(item.type)
        elif isinstance(item, six.string_types):
            item = item.lower()
            if item in ['cpu', 'none']:
                item = None
            elif item.startswith(('gpu', 'cuda')):
                item = item.replace('gpu', '')
                item = item.replace('cuda', '')
                if item == '':
                    item = 0
                elif ',' in item:
                    item = list(map(int, item.split(',')))
                else:
                    item = int(item)
            else:
                raise KeyError('Unknown XPU string coding')

        if check:
            if not XPU.exists(item):
                if isinstance(item, int) and not torch.cuda.is_available():
                    raise ValueError('XPU {} does not exist. '
                                     'CUDA is not available'.format(item))
                else:
                    raise ValueError('XPU {} does not exist.'.format(item))

        if item is None:
            xpu._main_device_id = None
            xpu._device_ids = None
        elif isinstance(item, six.integer_types):
            item = int(item)
            xpu._main_device_id = item
            xpu._device_ids = [item]
        elif isinstance(item, (list, tuple)):
            xpu._device_ids = list(item)
            if len(xpu._device_ids) == 0:
                raise IndexError('empty device list')
            xpu._main_device_id = xpu._device_ids[0]
        else:
            raise TypeError(xpu)

        if xpu._main_device_id is None:
            xpu.mode = 'cpu'
        else:
            if xpu._device_ids and len(xpu._device_ids) > 1:
                xpu.mode = 'multi-gpu'
            else:
                xpu.mode = 'gpu'

        if xpu._main_device_id is not None:
            xpu._cuda_device = torch.cuda.device(xpu._main_device_id)

    @classmethod
    def from_auto(XPU, min_memory=NETHARN_MIN_MB):
        """
        Determines what a CPU/GPU device to use based on hardware usage.

        Args:
            min_memory (int): min memory needed in bytes to default to GPU.
                defaults to envvar NETHARN_MIN_MB or 6000
        """
        if torch.cuda.is_available():
            n_available = torch.cuda.device_count()
            gpu_num = find_unused_gpu(min_memory=min_memory)
            if gpu_num is None or gpu_num >= n_available:
                gpu_num = None
        else:
            gpu_num = None
        xpu = XPU(gpu_num)
        return xpu

    @classmethod
    def from_argv(XPU, check=True, **kwargs):
        """
        Determine what CPU/GPU device to use based on sys.argv

        CommandLine:
            python -m netharn.device XPU.from_argv --gpu=0,1

        Example:
            >>> xpu = XPU.from_argv()
            >>> print(xpu)
        """
        item = ub.argval('--xpu', default=ub.NoParam)
        if item is not ub.NoParam:
            xpu = XPU.coerce(item)
        else:
            anygpu = ub.argflag('--gpu')
            if anygpu:
                gpu_num = XPU.default_gpu()
            else:
                gpu_num = ub.argval('--gpu', default=None)
            if ub.argflag('--cpu'):
                xpu = XPU(None, check=check)
            elif gpu_num is None:
                xpu = XPU.from_auto(**kwargs)
            else:
                if gpu_num.lower() == 'none':
                    xpu = XPU(None)
                if isinstance(gpu_num, six.string_types) and ',' in gpu_num:
                    _device_ids = list(map(int, gpu_num.split(',')))
                    xpu = XPU(_device_ids, check=check)
                else:
                    xpu = XPU(int(gpu_num), check=check)
        return xpu

    @classmethod
    def from_data(XPU, item, check=True, **kwargs):
        """
        Creates an XPU to represent the processing device(s) a Module, Tensor,
        or Variable currently exists on.

        Example:
            >>> xpu = XPU.from_data(torch.randn(3))
            >>> assert not xpu.is_gpu()
            >>> if torch.cuda.is_available():
            >>>     xpu = XPU.from_data(torch.randn(3).to('cuda'))
            >>>     assert xpu.is_gpu()
            >>>     for i in range(torch.cuda.device_count()):
            >>>         xpu = XPU.from_data(torch.randn(3).to(i))
            >>>         assert xpu.is_gpu()
            >>>         assert xpu._main_device_id == i
        """
        if hasattr(item, 'device'):
            return XPU(item.device, check=check)
        if hasattr(item, 'is_cuda'):
            if item.is_cuda:
                return XPU(item.get_device(), check=check)
            else:
                return XPU(None, check=check)
        elif hasattr(item, 'state_dict'):
            devices = [item.device for item in item.state_dict().values()]
            _device_ids = set()
            for device in devices:
                if device.type == 'cuda':
                    index = device.index or 0
                    _device_ids.add(index)
                else:
                    _device_ids.add(None)
            try:
                _device_ids = sorted(_device_ids)
            except TypeError:
                raise Exception('cannot currently mix CPU and GPU')
            return XPU(_device_ids, check=check)
        else:
            raise TypeError(type(item))

    of = from_data  # alias

    @classmethod
    def coerce(XPU, item, check=True, **kwargs):
        """
        Converts objects of many different types into an XPU.

        Args:
            item : special string, int, list, or None

        Example:
            >>> assert XPU.coerce('0', check=False) == XPU(0, check=False)
            >>> assert XPU.coerce('0,1,2', check=False) == XPU([0, 1, 2], check=False)
            >>> assert XPU.coerce('2,3,4', check=False) == XPU([2, 3, 4], check=False)
            >>> assert XPU.coerce('gpus=2,3,4', check=False) == XPU([2, 3, 4], check=False)
            >>> assert XPU.coerce([0, 1], check=False) == XPU([0, 1], check=False)
            >>> assert XPU.coerce(torch.Tensor()) == XPU(None)
            >>> assert XPU.coerce(None) == XPU(None)
            >>> assert XPU.coerce('auto', check=False) is not None
        """
        try:
            if item is None:
                return XPU(item, check=check)
            elif isinstance(item, XPU):
                return item
            elif isinstance(item, _TENSOR_TYPES):
                return XPU.from_data(item, check=check)
            elif isinstance(item, torch.nn.Module):
                return XPU.from_data(item, check=check)
            elif isinstance(item, six.integer_types):
                return XPU(int(item), check=check)
            elif isinstance(item, (list, tuple)):
                return XPU(item, check=check)
            elif isinstance(item, six.string_types):
                if item == 'auto':
                    return XPU.from_auto(**kwargs)
                elif item == 'argv':
                    return XPU.from_argv(check=check, **kwargs)
                if item == 'cpu' or item is None:
                    return XPU(None, check=check)
                elif item == 'cpu' or item is None:
                    return XPU(None, check=check)
                else:
                    item = item.lower()
                    item = item.replace('=', '')
                    item = item.replace('cpu', '')
                    item = item.replace('gpus', '')
                    item = item.replace('gpu', '')
                    item = item.replace('cuda', '')
                    if item == '':
                        if torch.cuda.is_available():
                            item = XPU.default_gpu()
                        else:
                            item = None
                    elif ',' in item:
                        item = list(map(int, item.split(',')))
                    elif item == 'none':
                        item = None
                    else:
                        item = int(item)
                    return XPU(item, check=check)
            else:
                ValueError
        except Exception as ex:
            raise ValueError('cannot cast to XPU. item={!r}. Caused by: {!r}'.format(item, ex))

    @classmethod
    def cast(xpu, item, check=True, **kwargs):
        """
        Deprecated, use XPU.coerce instead.
        """
        return xpu.coerce(item, **kwargs)

    def __eq__(xpu, other):
        """
        Example:
            >>> assert XPU([0], check=False) == XPU(0, check=False)
            >>> assert XPU('gpu', check=False) == XPU(0, check=False)
            >>> assert XPU([1], check=False) != XPU(0, check=False)
            >>> assert XPU('cpu', check=False) == XPU(None, check=False)
            >>> assert XPU([0, 1], check=False) == XPU([0, 1], check=False)
            >>> assert XPU([0, 1], check=False) != XPU([1, 0], check=False)
            >>> assert 'numpy' != XPU([1, 0], check=False)
        """
        try:
            return (xpu._main_device_id == other._main_device_id and
                    xpu._device_ids == other._device_ids)
        except AttributeError:
            return False

    @property
    def devices(xpu):
        """ A list of torch devices represented by this XPU """
        if xpu.is_gpu():
            return [torch.device(type='cuda', index=index)
                    for index in xpu._device_ids]
        else:
            return [torch.device(type='cpu')]

    @property
    def device(xpu):
        """ alias for main_device """
        return xpu.main_device

    @property
    def main_device(xpu):
        """
        The main torch device represented by this XPU

        Example:
            >>> xpu = XPU(None)
            >>> print(repr(xpu.main_device))
            device(type='cpu')
        """
        if xpu.is_gpu():
            return torch.device(type='cuda', index=xpu._main_device_id)
        else:
            return torch.device(type='cpu')

    @classmethod
    def exists(XPU, item):
        """
        Determins if GPU/CPU exists

        Args:
            item (int or None):
        """
        if item is None:
            return True
        elif isinstance(item, six.integer_types):
            if item < 0:
                raise ValueError('gpu num must be positive not {}'.format(item))
            return (torch.cuda.is_available() and
                    item < torch.cuda.device_count())
        elif isinstance(item, (tuple, list)):
            return all(XPU.exists(i) for i in item)
        else:
            raise TypeError(type(item))

    def memory(self):
        """
        Example:
            >>> from netharn.device import *
            >>> print(ub.repr2(XPU.coerce(None).memory()))
            {
                'available': ...,
                'total': ...,
                'used': ...,
            }
            >>> # xdoctest: +REQUIRES(--cuda)
            >>> print(ub.repr2(XPU.coerce(0).memory()))
            {
                'available': ...,
                'total': ...,
                'used': ...,
            }

        """
        gpus = gpu_info()
        info = {
            'available': 0,
            'total': 0,
            'used': 0,
        }
        if self._device_ids is None:
            import psutil
            tup = psutil.virtual_memory()
            MB = 1 / 2 ** 20
            info['total'] += tup.total * MB
            info['used'] += tup.used * MB
            info['available'] += tup.available * MB
        else:
            for id in self._device_ids:
                info['total'] += gpus[id]['mem_total']
                info['used'] += gpus[id]['mem_used']
                info['available'] += gpus[id]['mem_avail']
        return info

    def __str__(xpu):
        return xpu.__nice__()

    def __enter__(xpu):
        if xpu._cuda_device:
            xpu._cuda_device.__enter__()
        return xpu

    def __exit__(xpu, ex_type, ex_value, tb):
        if xpu._cuda_device:
            return xpu._cuda_device.__exit__(ex_type, ex_value, tb)

    def __nice__(xpu):
        if xpu.is_gpu():
            if xpu._device_ids and len(xpu._device_ids) > 1:
                parts = [str(n) + '*' if n == xpu._main_device_id else str(n)
                         for n in xpu._device_ids]
                return 'GPU({})'.format(','.join(parts))
            else:
                return 'GPU({})'.format(xpu._main_device_id)
        else:
            return 'CPU'

    def __json__(xpu):
        """
        String encoding of an XPU

        CommandLine:
            xdoctest -m ~/code/netharn/netharn/device.py XPU.__json__

        Example:
            >>> print(XPU(None).__json__())
            CPU
            >>> print(XPU(0, check=False).__json__())
            GPU(0)
            >>> print(XPU([1, 2, 3], check=False).__json__())
            GPU(1*,2,3)
        """
        return str(xpu)

    def __int__(xpu):
        return xpu._main_device_id

    def number_of_devices(xpu):
        """ The number of underlying devices abstracted by this XPU """
        return 1 if not xpu._device_ids else len(xpu._device_ids)

    def is_cpu(xpu):
        return xpu._main_device_id is None

    def is_gpu(xpu):
        """ True if running in single or parallel gpu mode """
        return xpu._main_device_id is not None
        # return 'gpu' in xpu.mode

    @staticmethod
    def raw(model):
        """
        Unmounts the original core model if it is mounted.

        Args:
            model (torch.nn.Module): a model (potentially mounted)

        Returns:
            torch.nn.Module:
                if `model` is mounted returns `model.module`
                otherwise, returns `model`
        """
        if isinstance(model, (MountedModel, torch.nn.DataParallel)):
            # Unwrap the core model
            model = model.module
        return model

    def mount(xpu, model):
        """
        Like move, but only for models.
        Note that this works inplace for non-Tensor objects.

        Args:
            model (torch.nn.Module): the model to mount

        Returns:
            DataSerial | DataParallel :
                the model mounted on the XPU (which may be multiple GPUs)

        Example:
            >>> model = torch.nn.Conv2d(1, 1, 1)
            >>> xpu = XPU()
        """
        # Unwrap the core model if necessary
        model = xpu.raw(model)
        model = xpu.move(model)
        if xpu._device_ids and len(xpu._device_ids) > 1:
            model = DataParallel(model, device_ids=xpu._device_ids,
                                 output_device=xpu._main_device_id)
        else:
            model = DataSerial(model)
        return model

    def move(xpu, data, **kwargs):
        """
        Moves the model onto the primary GPU or CPU.

        If the data is nested in a container (e.g. a dict or list) then this
        funciton is applied recursively to all values in the container.

        Note:
            This works by calling the `.to` method, which works inplace for
            torch Modules, but is not implace for raw Tensors.

        Args:
            data (torch.Module | torch.Tensor | Collection):
                raw data or a collection containing raw data.
            **kwargs : forwarded to `data.cuda`

        Returns:
            torch.Tensor: the tensor with a dtype for this device

        Example:
            >>> data = torch.FloatTensor([0])
            >>> if torch.cuda.is_available():
            >>>     xpu = XPU.coerce('gpu')
            >>>     assert isinstance(xpu.move(data), torch.cuda.FloatTensor)
            >>> xpu = XPU.coerce('cpu')
            >>> assert isinstance(xpu.move(data), torch.FloatTensor)
            >>> assert isinstance(xpu.move([data])[0], torch.FloatTensor)
            >>> assert isinstance(xpu.move({0: data})[0], torch.FloatTensor)
            >>> assert isinstance(xpu.move({data}), set)
        """
        try:
            if xpu.is_gpu():
                return data.to(xpu._main_device_id, **kwargs)
            else:
                return data.to('cpu')
        except AttributeError:
            # Recursive move
            if isinstance(data, container_abcs.Mapping):
                cls = data.__class__
                return cls((k, xpu.move(v)) for k, v in data.items())
            elif isinstance(data, (container_abcs.Sequence, container_abcs.Set)):
                cls = data.__class__
                return cls(xpu.move(v) for v in data)
            else:
                raise TypeError('Unknown type {}'.format(type(data)))

    @classmethod
    def default_gpu(XPU):
        """
        Example:
            >>> print(XPU.default_gpu())
        """
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        else:
            return None

    def set_as_default(xpu):
        """
        Sets this device as the default torch GPU

        Example:
            >>> import pytest
            >>> XPU(None).set_as_default()
            >>> if torch.cuda.is_available():
            >>>     XPU(0).set_as_default()
            >>>     assert torch.cuda.current_device() == 0
        """
        if xpu.is_gpu():
            torch.cuda.set_device(xpu._main_device_id)
        else:
            torch.cuda.set_device(-1)

    def load(xpu, fpath):
        """
        Loads data from a filepath onto this XPU

        Args:
            fpath (str or file): path to torch data file or file-like object

        Example:
            >>> from os.path import join
            >>> dpath = ub.ensure_app_cache_dir('netharn')
            >>> fpath = join(dpath, 'foo.pt')
            >>> cpu = XPU(None)
            >>> data = torch.FloatTensor([0])
            >>> torch.save(data, fpath)
            >>> loaded = cpu.load(fpath)
            >>> assert all(data == loaded)
        """
        # print('Loading data onto {} from {}'.format(xpu, fpath))
        try:
            return torch.load(fpath, map_location=xpu._map_location)
        except Exception:
            print('XPU={} Failed to load fpath={}'.format(xpu, fpath))
            raise

    def _map_location(xpu, storage, location):
        """
        Helper for `xpu.load` used when calling `torch.load`

        Args:
            storage (torch.Storage) : the initial deserialization of the
                storage of the data read by `torch.load`, residing on the CPU.
            location (str): tag identifiying the location the data being read
                by `torch.load` was originally saved from.

        Returns:
            torch.Storage : the storage
        """
        if xpu.is_gpu():
            return storage.cuda(xpu._main_device_id)
        else:
            return storage

    def synchronize(xpu):
        """
        Should be used when benchmarking performance of GPU implementaions
        """
        if xpu.is_gpu():
            torch.cuda.synchronize()


def find_unused_gpu(min_memory=0):
    """
    Finds GPU with the lowest memory usage by parsing output of nvidia-smi

    Args:
        min_memory (int): disregards GPUs with fewer than `min_memory` free MB

    Returns:
        int or None: gpu num if a match is found otherwise None

    CommandLine:
        python -c "from netharn import device; print(device.find_unused_gpu(300))"

    Example:
        >>> if torch.cuda.is_available():
        >>>     item = find_unused_gpu()
        >>>     assert item is None or isinstance(item, int)
    """
    gpus = gpu_info()
    if not gpus:
        return None

    # Order GPUs by most available memory
    # gpu_avail_mem = {n: -gpu['mem_avail'] for n, gpu in gpus.items()}

    # Order GPUs by fewest compute processes, and then by available memory
    gpu_avail_mem = {n: (gpu['num_compute_procs'], -gpu['mem_avail'])
                     for n, gpu in gpus.items()}
    ranked_order = ub.argsort(gpu_avail_mem)

    for gpu_num in ranked_order:
        gpu = gpus[gpu_num]
        if gpu['mem_avail'] >= min_memory:
            return gpu_num
    return None


def gpu_info(new_mode=False):
    """
    Run nvidia-smi and parse output

    Returns:
        OrderedDict: info about each GPU indexed by gpu number

    Note:
        Does not gaurentee CUDA is installed.

    Warnings:
        if nvidia-smi is not installed

    CommandLine:
        xdoctest -m netharn.device gpu_info --cuda

    Example:
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from netharn.device import *
        >>> gpus = gpu_info()
        >>> print('gpus = {}'.format(ub.repr2(gpus, nl=3)))
        >>> assert len(gpus) == torch.cuda.device_count()
    """
    pass

    """
    Ignore:
        # TODO: make more efficient calls to nvidia-smi

        utilization.gpu
        utilization.memory
        compute_mode
        memory.total
        memory.used
        memory.free
        index
        name
        count

        nvidia-smi  -h
        nvidia-smi  --help-query-compute-apps
        nvidia-smi  --help-query-gpu

        nvidia-smi --help-query-accounted-apps
        nvidia-smi --help-query-supported-clocks
        nvidia-smi --help-query-retired-pages
        nvidia-smi --query-accounted-apps="pid" --format=csv

        nvidia-smi  --query-gpu="index,memory.total,memory.used,memory.free,count,name,gpu_uuid" --format=csv
        nvidia-smi  --query-compute-apps="pid,name,gpu_uuid,used_memory" --format=csv
    """

    if new_mode:

        def _query_nvidia_smi(mode, fields):
            """
            Runs nvidia smi in query mode

            Args:
                mode (str): the query cli flag to pass to nvidia-smi
                fields (List[str]): csv header fields to query

            Returns:
                List[Dict[str, str]]: parsed csv output
            """
            header = ','.join(fields)
            cmd_fmtstr = 'nvidia-smi --{mode}="{header}" --format=csv,noheader'
            info = ub.cmd(cmd_fmtstr.format(mode=mode, header=header))
            if info['ret'] != 0:
                raise Exception('unable to call nvidia-smi')
            rows = []
            for line in info['out'].split('\n'):
                line = line.strip()
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    row = ub.dzip(fields, parts)
                    rows.append(row)
            return rows

        fields = ['index', 'memory.total', 'memory.used', 'memory.free',
                  'count', 'name', 'gpu_uuid']
        mode = 'query-gpu'
        gpu_rows = _query_nvidia_smi(mode, fields)

        fields = ['pid', 'name', 'gpu_uuid', 'used_memory']
        mode = 'query-compute-apps'
        proc_rows = _query_nvidia_smi(mode, fields)

        # Coerce into the old-style format for backwards compatibility
        gpus = {}
        for row in gpu_rows:
            gpu = row.copy()
            num = int(gpu['index'])
            gpu['num'] = num
            gpu['mem_used'] = float(gpu['memory.used'].strip().replace('MiB', ''))
            gpu['mem_total'] = float(gpu['memory.total'].strip().replace('MiB', ''))
            gpu['mem_avail'] = gpu['mem_total'] - gpu['mem_used']
            gpu['procs'] = []
            gpus[num] = gpu

        gpu_uuid_to_num = {g['gpu_uuid']: gpu['num'] for g in gpus.values()}

        for row in proc_rows:
            # Give each GPU info on which processes are using it
            proc = row.copy()
            proc['type'] = 'C'
            proc['gpu_num'] = gpu_uuid_to_num[proc['gpu_uuid']]
            num = proc['gpu_num']
            gpus[num]['procs'].append(proc)

        for gpu in gpus.values():
            # Let each GPU know how many processes are currently using it
            num_compute_procs = 0
            # num_graphics_procs = 0
            for proc in gpu['procs']:
                if proc['type'] == 'C':
                    num_compute_procs += 1
                # elif proc['type'] == 'G':
                #     num_graphics_procs += 1
                else:
                    raise NotImplementedError(proc['type'])

            # NOTE calling nvidia-smi in query mode does not seem to have
            # support for getting info about graphics procs.
            gpu['num_compute_procs'] = num_compute_procs
            # gpu['num_graphics_procs'] = num_graphics_procs
    else:
        try:
            result = ub.cmd('nvidia-smi')
            if result['ret'] != 0:
                warnings.warn('Problem running nvidia-smi.')
                return None
        except Exception:
            warnings.warn('Could not run nvidia-smi.')
            return {}

        lines = result['out'].splitlines()

        gpu_lines = []
        proc_lines = []
        current = None

        state = '0_gpu_read'

        for line in lines:
            if current is None:
                # Signals the start of GPU info
                if line.startswith('|====='):
                    current = []
            else:
                if state == '0_gpu_read':
                    if len(line.strip()) == 0:
                        # End of GPU info
                        state = '1_proc_read'
                        current = None
                    elif line.startswith('+----'):
                        # Move to the next GPU
                        gpu_lines.append(current)
                        current = []
                    else:
                        current.append(line)
                elif state == '1_proc_read':
                    if line.startswith('+----'):
                        # Move to the next GPU
                        # End of proc info
                        state = 'terminate'
                        break
                    else:
                        proc_lines.append(line)
                else:
                    raise AssertionError(state)

        def parse_gpu_lines(lines):
            line1 = lines[0]
            line2 = lines[1]
            gpu = {}
            gpu['name'] = ' '.join(line1.split('|')[1].split()[1:-1])
            gpu['num'] = int(' '.join(line1.split('|')[1].split()[0]))

            mempart = line2.split('|')[2].strip()
            part1, part2 = mempart.split('/')
            gpu['mem_used'] = float(part1.strip().replace('MiB', ''))
            gpu['mem_total'] = float(part2.strip().replace('MiB', ''))
            gpu['mem_avail'] = gpu['mem_total'] - gpu['mem_used']
            return gpu

        def parse_proc_line(line):
            inner = '|'.join(line.split('|')[1:-1])
            parts = [p.strip() for p in inner.split(' ')]
            parts = [p for p in parts if p]

            index = int(parts[0])
            pid = int(parts[1])
            proc_type = str(parts[2])
            proc_name = str(parts[3])
            used_mem = float(parts[4].replace('MiB', ''))

            proc = {
                'gpu_num': index,
                'pid': pid,
                'type': proc_type,
                'name': proc_name,
                'used_mem': used_mem,
            }
            return proc

        gpus = {}
        for num, lines in enumerate(gpu_lines):
            gpu = parse_gpu_lines(lines)
            assert num == gpu['num'], (
                'nums ({}, {}) do not agree. probably a parsing error'.format(num, gpu['num']))
            assert num not in gpus, (
                'Multiple GPUs labeled as num {}. Probably a parsing error'.format(num))
            gpus[num] = gpu
            gpus[num]['procs'] = []

        for line in proc_lines:
            # Give each GPU info on which processes are using it
            proc = parse_proc_line(line)
            num = proc['gpu_num']
            gpus[num]['procs'].append(proc)

        for gpu in gpus.values():
            # Let each GPU know how many processes are currently using it
            num_compute_procs = 0
            num_graphics_procs = 0
            for proc in gpu['procs']:
                if proc['type'] == 'C':
                    num_compute_procs += 1
                elif proc['type'] == 'G':
                    num_graphics_procs += 1
                else:
                    raise NotImplementedError(proc['type'])
            gpu['num_compute_procs'] = num_compute_procs
            gpu['num_graphics_procs'] = num_graphics_procs

    return gpus


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.device all
        pytest ~/code/netharn/netharn/device.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
