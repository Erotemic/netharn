#!/usr/bin/env python
# -*- coding: utf-8 -*-
# NOTE: pip install -U --pre h5py
"""
Installation:
    pip install git+https://github.com/Erotemic/netharn.git

Developing:
    git clone https://github.com/Erotemic/netharn.git
    pip install -e netharn

Pypi:
     # Presetup
     pip install twine

     # First tag the source-code
     VERSION=$(python -c "import setup; print(setup.version)")
     echo $VERSION
     git tag $VERSION -m "tarball tag $VERSION"
     git push --tags origin master

     # NEW API TO UPLOAD TO PYPI
     # https://packaging.python.org/tutorials/distributing-packages/

     # Build wheel or source distribution
     python setup.py bdist_wheel --universal

     # Use twine to upload. This will prompt for username and password
     twine upload --username erotemic --skip-existing dist/*

     # Check the url to make sure everything worked
     https://pypi.org/project/netharn/

     # ---------- OLD ----------------
     # Check the url to make sure everything worked
     https://pypi.python.org/pypi?:action=display&name=netharn

"""
from setuptools import setup, find_packages
import os
import sys
import numpy as np
from os.path import dirname, join, exists, abspath
from distutils.extension import Extension
from Cython.Distutils import build_ext


def parse_version(package):
    """
    Statically parse the version number from __init__.py

    CommandLine:
        python -c "import setup; print(setup.parse_version('netharn'))"
    """
    import ast
    init_fpath = join(dirname(__file__), package, '__init__.py')
    with open(init_fpath) as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if target.id == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.md')
    # print('readme_fpath = %r' % (readme_fpath,))
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        # try:
        #     # convert markdown to rst for pypi
        #     import pypandoc
        #     return pypandoc.convert(readme_fpath, 'rst')
        # except Exception as ex:
            # strip out markdown to make a clean readme for pypi
            textlines = []
            with open(readme_fpath, 'r') as f:
                capture = False
                for line in f.readlines():
                    if '# Purpose' in line:
                        capture = True
                    elif line.startswith('##'):
                        break
                    elif capture:
                        textlines += [line]
            text = ''.join(textlines).strip()
            text = text.replace('\n\n', '_NLHACK_')
            text = text.replace('\n', ' ')
            text = text.replace('_NLHACK_', '\n\n')
            return text


def parse_requirements(fname='requirements.txt'):
    """
    Parse the package dependencies listed in a requirements file.

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    from os.path import dirname, join, exists
    require_fpath = join(dirname(__file__), fname)
    # This breaks on pip install, so check that it exists.
    if exists(require_fpath):
        with open(require_fpath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [line for line in lines if not line.startswith('#')]
            return lines
    return []


def conditional_requirements(pkgmods):
    # ONLY require these if the packages aren't installed because python
    # doesn't seem to recognize that if we already have them.
    for pkgname, modname in pkgmods:
        try:
            __import__(modname)
        except ImportError:
            yield pkgname


def find_in_path(name, path):
    """
    Find a file in a search path
    adapted fom
    http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    """
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # there doesnt seem to be an accepted standard for the CUDA envvar yet
    cuda_environs = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_SDK_ROOT_DIR', 'CUDAHOME']
    cuda_environs = [key for key in cuda_environs if key in os.environ]

    # first check for the env variable CUDA_HOME / CUDA_ROOT / etc.
    if cuda_environs:
        cuda_environ = cuda_environs[0]
        home = os.environ[cuda_environ]
        nvcc = join(home, 'bin', 'nvcc')
        if not exists(nvcc):
            raise EnvironmentError(
                'The nvcc binary={} does not exist in ${}'.format(
                    nvcc, cuda_environ))
    else:
        # otherwise, search the PATH for NVCC
        default_path = join(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc',
                            os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. '
                                   'Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': join(home, 'include'),
                  'lib64': join(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not '
                                   'be located in %s' % (k, v))

    return cudaconfig


CUDACONFIG = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDACONFIG['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# --------------------------------------------------------
# Some code was derived from Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
ext_modules = []


yolo_util_m = 'netharn.models.yolo2.utils.'
yolo_util_p = yolo_util_m.replace('.', os.path.sep)

ext_modules += [
    Extension(
        yolo_util_m + "cython_bbox",
        [join(yolo_util_p, "cython_bbox.pyx")],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        yolo_util_m + "cython_yolo",
        [join(yolo_util_p, "cython_yolo.pyx")],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        yolo_util_m + "nms.cpu_nms",
        [join(yolo_util_p, "nms/cpu_nms.pyx")],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
]

# ext_modules += [
#     Extension(
#         yolo_util_m + 'pycocotools._mask',
#         sources=[
#             join(yolo_util_p, 'pycocotools/maskApi.c'),
#             join(yolo_util_p, 'pycocotools/_mask.pyx'),
#         ],
#         include_dirs=[numpy_include, join(yolo_util_p, 'pycocotools')],
#         extra_compile_args={
#             'gcc': ['-Wno-cpp', '-Wno-unused-function', '-std=c99']},
#     ),
# ]

ext_modules += [
    Extension(yolo_util_m + 'nms.gpu_nms',
              [join(yolo_util_p, 'nms/nms_kernel.cu'),
               join(yolo_util_p, 'nms/gpu_nms.pyx')],
              library_dirs=[CUDACONFIG['lib64']],
              libraries=['cudart'],
              language='c++',
              runtime_library_dirs=[CUDACONFIG['lib64']],
              # this syntax is specific to this build system
              # we're only going to use certain compiler args with
              # nvcc and not with gcc
              # the implementation of this trick is in
              # customize_compiler() below
              extra_compile_args={'gcc': ["-Wno-unused-function"],
                                  'nvcc': ['-arch=sm_35',
                                           '--ptxas-options=-v',
                                           '-c',
                                           '--compiler-options',
                                           "'-fPIC'"]},
              include_dirs=[numpy_include, CUDACONFIG['include']]
              ),
]


def clean():
    """
    __file__ = ub.truepath('~/code/netharn/setup.py')
    """
    import ubelt as ub
    import os

    modname = 'netharn'
    repodir = dirname(__file__)
    # pkgdir = join(repodir, modname)

    toremove = []
    for root, dnames, fnames in os.walk(repodir):

        if os.path.basename(root) == modname + '.egg-info':
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == '__pycache__':
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == '_ext':
            # Remove torch extensions
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == 'build':
            # Remove python c extensions
            if len(dnames) == 1 and dnames[0].startswith('temp.'):
                toremove.append(root)
                del dnames[:]

        # Remove simple pyx inplace extensions
        for fname in fnames:
            if fname.endswith('.so') or fname.endswith('.c'):
                if fname.split('.')[0] + '.pyx' in fnames:
                    toremove.append(join(root, fname))

    def enqueue(d):
        if exists(d) and d not in toremove:
            toremove.append(d)

    enqueue(join(repodir, 'netharn/models/yolo2/utils/cython_yolo.c') )
    enqueue(join(repodir, 'netharn/models/yolo2/utils/cython_bbox.c') )
    enqueue(join(repodir, 'netharn/utils/nms/cpu_nms.c') )
    enqueue(join(repodir, 'netharn/utils/nms/cpu_nms.c') )
    enqueue(join(repodir, 'netharn/utils/nms/cpu_nms.cpp') )

    enqueue(join(repodir, 'netharn/layers/roi_pooling/_ext') )
    enqueue(join(repodir, 'netharn/layers/reorg/_ext') )
    import glob

    for d in glob.glob(join(repodir, 'netharn/utils/nms/*_nms.*so')):
        enqueue(d)

    for dpath in toremove:
        # print('Removing dpath = {!r}'.format(dpath))
        ub.delete(dpath, verbose=1)


class TorchExtension(object):
    """
    customized hacked extension

    TODO: is there a better way to do this?
    """

    def __init__(self, name, sources, cuda_sources, extra_compile_args={},
                 with_cuda=True):
        self.name = name
        self.sources = sources
        self.cuda_sources = cuda_sources
        self.extra_compile_args = extra_compile_args
        self.with_cuda = with_cuda

    def build(self):
        from torch.utils.ffi import create_extension
        import ubelt as ub

        sources = [p for p in self.sources if p.endswith('.c')]
        headers = [p for p in self.sources if p.endswith('.h')]
        cu_sources = [p for p in self.cuda_sources if p.endswith('.cu')]

        extra_objects = []
        defines = []
        if self.with_cuda:
            sources += [p for p in self.cuda_sources if p.endswith('.c')]
            headers += [p for p in self.cuda_sources if p.endswith('.h')]
            cu_objects = [p + '.o' for p in cu_sources]

            extra = ' '.join(self.extra_compile_args.get('nvcc', []))
            command_fmt = '{nvcc_exe} -c -o {cu_objects} {cu_sources} {extra}'
            command = command_fmt.format(
                nvcc_exe=CUDACONFIG['nvcc'],
                cu_objects=' '.join(cu_objects),
                cu_sources=' '.join(cu_sources),
                extra=extra,
            )
            info = ub.cmd(command, verbout=1, verbose=2)
            if info['ret'] != 0:
                raise Exception('Failed to build extension ' + self.name)

            for fpath in cu_objects:
                if not exists(fpath):
                    raise Exception('Object {} does not exist'.format(fpath))

            extra_objects += [abspath(p) for p in cu_objects]
            defines += [('WITH_CUDA', None)]

        ffi = create_extension(
            self.name,
            headers=headers,
            sources=sources,
            define_macros=defines,
            relative_to=__file__,
            with_cuda=self.with_cuda,
            extra_objects=extra_objects,
            # extra_compile_args=self.extra_compile_args
        )
        ffi.build()

yolo_layers_m = 'netharn.models.yolo2.layers.'
yolo_layers_p = yolo_layers_m.replace('.', os.path.sep)

# Is there a better way to incorporate these into normal ext_modules
reorg_ext = TorchExtension(
    yolo_layers_m + 'reorg._ext.reorg_layer',
    sources=[
        join(yolo_layers_p, 'reorg/src/reorg_cpu.c'),
        join(yolo_layers_p, 'reorg/src/reorg_cpu.h')
    ],
    cuda_sources=[
        join(yolo_layers_p, 'reorg/src/reorg_cuda_kernel.cu'),
        join(yolo_layers_p, 'reorg/src/reorg_cuda.c'),
        join(yolo_layers_p, 'reorg/src/reorg_cuda.h')
    ],
    extra_compile_args={
        'gcc': ["-Wno-unused-function"],
        'nvcc': '-x cu -Xcompiler -fPIC -arch=sm_52'.split(' ')
    }
)

# reorg_ext.build()
# from netharn.models.yolo2.layers.reorg import reorg_layer
# from netharn.models.yolo2.layers.reorg._ext.reorg_layer import _reorg_layer
# print('reorg_layer = {!r}'.format(dir(reorg_layer)))
# print('reorg_layer.reorg_layer = {!r}'.format(dir(reorg_layer.reorg_layer)))
# import ubelt as ub
# print('_reorg_layer = {}'.format(ub.repr2(dir(_reorg_layer.lib))))
# sys.exit(0)

# roi_ext = TorchExtension(
#     yolo_layers_m + 'roi_pooling._ext.roi_pooling',
#     sources=[
#         join(yolo_layers_p, 'roi_pooling/src/roi_pooling.c'),
#         join(yolo_layers_p, 'roi_pooling/src/roi_pooling.h')
#     ],
#     cuda_sources=[
#         join(yolo_layers_p, 'roi_pooling/src/cuda/roi_pooling_kernel.cu'),
#         join(yolo_layers_p, 'roi_pooling/src/roi_pooling_cuda.c'),
#         join(yolo_layers_p, 'roi_pooling/src/roi_pooling_cuda.h')
#     ],
#     extra_compile_args={
#         'gcc': ["-Wno-unused-function"],
#         'nvcc': '-x cu -Xcompiler -fPIC -arch=sm_52'.split(' ')
#     }
# )

torch_ffi_ext_modules = [reorg_ext]


if __name__ == '__main__':

    requirements = parse_requirements('requirements.txt')
    requirements += list(conditional_requirements([
        ('opencv_python', 'cv2'),
        ('pytorch', 'torch'),
    ]))

    if 'clean' in sys.argv:
        # hack
        clean()
        sys.exit(0)

    if 'build_ext' in sys.argv:
        # hack hack hack
        for ext in torch_ffi_ext_modules:
            ext.build()

    setup(
        name='netharn',
        version=parse_version('netharn'),
        author='Jon Crall',
        description='',
        long_description=parse_description(),
        install_requires=requirements,
        extras_require={
            'all': parse_requirements('optional-requirements.txt')
        },
        author_email='erotemic@gmail.com',
        url='https://github.com/Erotemic/netharn',
        license='Apache 2',
        packages=find_packages(),

        # inject our custom nvcc trigger
        cmdclass={'build_ext': custom_build_ext},
        ext_modules=ext_modules,

        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Utilities',
            # 'Topic :: DeepLearning',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3.6',
        ],
    )
