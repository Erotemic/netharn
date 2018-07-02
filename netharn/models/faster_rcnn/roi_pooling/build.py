"""
References:
    https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/roi_pooling/build.py
"""
import os
import torch
from torch.utils.ffi import create_extension
from os.path import join, dirname, realpath


def main():
    base_dpath = dirname(realpath(__file__))
    print('base_dpath = {!r}'.format(base_dpath))

    sources = [join(base_dpath, 'src/roi_pooling.c')]
    headers = [join(base_dpath, 'src/roi_pooling.h')]
    defines = []
    extra_objects = []

    with_cuda = torch.cuda.is_available()
    if with_cuda:
        print('Including CUDA code.')
        sources += ['src/roi_pooling_cuda.c']
        headers += ['src/roi_pooling_cuda.h']
        defines += [('WITH_CUDA', None)]

        roi_pooling_input = join(base_dpath, 'src', 'roi_pooling_kernel.cu')
        roi_pooling_output =  join(base_dpath, 'src', 'roi_pooling_kernel.cu.o')

        print("Compiling roi pooling kernels by nvcc...")
        # Note: building the extra objects was done in an external makefile
        # https://github.com/longcw/faster_rcnn_pytorch/blob/master/faster_rcnn/make.sh
        flags = '-D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52'
        command = 'nvcc -c -o {output} {inputs} {flags}'.format(
            inputs=roi_pooling_input,
            output=roi_pooling_output,
            flags=flags)
        import ubelt as ub
        ub.cmd(command, verbout=1, verbose=2)
        extra_objects.append(roi_pooling_output)

    for fpath in sources + headers + extra_objects:
        assert os.path.exists(fpath), 'must exist {}'.format(fpath)

    ffi = create_extension(
        '_ext.roi_pooling',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        extra_objects=extra_objects,
    )
    ffi.build()

if __name__ == '__main__':
    """
    sudo apt-get install python3 python-dev python3-dev \

     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip

    python ~/code/netharn/netharn/models/faster_rcnn/roi_pooling/build.py
    """
    main()
