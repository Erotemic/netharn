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
     python setup.py bdist_wheel --py-limited-api=cp36

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
from os.path import dirname, join, exists
from distutils.extension import Extension

try:
    import numpy as np
    from Cython.Distutils import build_ext
except ImportError:
    print('''
          Please Run:
              pip install numpy Cython
          ''')
    raise ImportError(
        'Numpy and Cython must be installed before you run setup.py. '
        'Please send a PR if you know how to fix this.')
import gpu_setup


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

version = parse_version('netharn')


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.md')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
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
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    from os.path import dirname, join, exists
    import re
    require_fpath = join(dirname(__file__), fname)
    # This breaks on pip install, so check that it exists.
    if exists(require_fpath):
        with open(require_fpath, 'r') as f:
            packages = []
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.startswith('-e '):
                        package = line.split('#egg=')[1]
                        packages.append(package)
                    else:
                        pat = '|'.join(['>', '>=', '=='])
                        package = re.split(pat, line)[0]
                        packages.append(package)
            return packages
    return []


def conditional_requirements(pkgmods):
    # ONLY require these if the packages aren't installed because python
    # doesn't seem to recognize that if we already have them.
    for pkgname, modname in pkgmods:
        try:
            __import__(modname)
        except ImportError:
            yield pkgname


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

    enqueue(join(repodir, 'netharn/util/nms/cpu_nms.c') )
    enqueue(join(repodir, 'netharn/util/nms/cpu_nms.c') )
    enqueue(join(repodir, 'netharn/util/nms/cpu_nms.cpp') )
    enqueue(join(repodir, 'netharn/util/nms/cython_boxes.c') )
    enqueue(join(repodir, 'netharn/util/nms/cython_boxes.html') )

    enqueue(join(repodir, 'netharn/layers/roi_pooling/_ext') )
    enqueue(join(repodir, 'netharn/layers/reorg/_ext') )
    import glob

    for d in glob.glob(join(repodir, 'netharn/util/nms/*_nms.*so')):
        enqueue(d)

    for d in glob.glob(join(repodir, 'netharn/util/nms/cython_boxes*.*so')):
        enqueue(d)

    for dpath in toremove:
        # print('Removing dpath = {!r}'.format(dpath))
        ub.delete(dpath, verbose=1)


try:
    gpu_setup.locate_cuda()
    DO_COMPILE = True
except EnvironmentError:
    print('Cant locate cuda. Skipping GPU build')
    # probably dont need to skip EVERYTHING
    DO_COMPILE = False


ext_modules = []
if DO_COMPILE:
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    # --------------------------------------------------------
    # Some code was derived from Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------

    util_m = 'netharn.util.'
    util_p = util_m.replace('.', os.path.sep)

    ext_modules += [
        Extension(
            util_m + "cython_boxes",
            [join(util_p, "cython_boxes.pyx")],
            extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
            include_dirs=[numpy_include]
        ),
        # Extension(
        #     util_m + "cython_yolo",
        #     [join(util_p, "cython_yolo.pyx")],
        #     extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        #     include_dirs=[numpy_include]
        # ),
        Extension(
            util_m + "nms.cpu_nms",
            [join(util_p, "nms/cpu_nms.pyx")],
            extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
            include_dirs=[numpy_include]
        ),
    ]


    ext_modules += [
        Extension(util_m + 'nms.gpu_nms',
                  [join(util_p, 'nms/nms_kernel.cu'),
                   join(util_p, 'nms/gpu_nms.pyx')],
                  library_dirs=[gpu_setup.CUDACONFIG['lib64']],
                  libraries=['cudart'],
                  language='c++',
                  runtime_library_dirs=[gpu_setup.CUDACONFIG['lib64']],
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
                  include_dirs=[numpy_include, gpu_setup.CUDACONFIG['include']]
                  ),
    ]

# layers_m = 'netharn.layers.'
# layers_p = layers_m.replace('.', os.path.sep)

# # Is there a better way to incorporate these into normal ext_modules
# reorg_ext = TorchExtension(
#     layers_m + 'reorg._ext.reorg_layer',
#     sources=[
#         join(layers_p, 'reorg/src/reorg_cpu.c'),
#         join(layers_p, 'reorg/src/reorg_cpu.h')
#     ],
#     cuda_sources=[
#         join(layers_p, 'reorg/src/reorg_cuda_kernel.cu'),
#         join(layers_p, 'reorg/src/reorg_cuda.c'),
#         join(layers_p, 'reorg/src/reorg_cuda.h')
#     ],
#     extra_compile_args={
#         'gcc': ["-Wno-unused-function"],
#         'nvcc': '-x cu -Xcompiler -fPIC -arch=sm_52'.split(' ')
#     }
# )

# reorg_ext.build()
# from netharn.layers.reorg import reorg_layer
# from netharn.layers.reorg._ext.reorg_layer import _reorg_layer
# print('reorg_layer = {!r}'.format(dir(reorg_layer)))
# print('reorg_layer.reorg_layer = {!r}'.format(dir(reorg_layer.reorg_layer)))
# import ubelt as ub
# print('_reorg_layer = {}'.format(ub.repr2(dir(_reorg_layer.lib))))
# sys.exit(0)

# roi_ext = TorchExtension(
#     layers_m + 'roi_pooling._ext.roi_pooling',
#     sources=[
#         join(layers_p, 'roi_pooling/src/roi_pooling.c'),
#         join(layers_p, 'roi_pooling/src/roi_pooling.h')
#     ],
#     cuda_sources=[
#         join(layers_p, 'roi_pooling/src/cuda/roi_pooling_kernel.cu'),
#         join(layers_p, 'roi_pooling/src/roi_pooling_cuda.c'),
#         join(layers_p, 'roi_pooling/src/roi_pooling_cuda.h')
#     ],
#     extra_compile_args={
#         'gcc': ["-Wno-unused-function"],
#         'nvcc': '-x cu -Xcompiler -fPIC -arch=sm_52'.split(' ')
#     }
# )

# torch_ffi_ext_modules = [reorg_ext]

if DO_COMPILE:
    # run the customize_compiler
    class custom_build_ext(build_ext):
        def build_extensions(self):
            gpu_setup.customize_compiler_for_nvcc(self.compiler)
            build_ext.build_extensions(self)
    compile_setup_kw = dict(
        cmdclass={'build_ext': custom_build_ext},
        ext_modules=ext_modules,
    )
else:
    compile_setup_kw = {}


if __name__ == '__main__':

    requirements = parse_requirements('requirements.txt')
    requirements += list(conditional_requirements([
        ('opencv_python', 'cv2'),
        ('torch', 'torch'),
    ]))

    if 'clean' in sys.argv:
        # hack
        clean()
        sys.exit(0)

    # if 'build_ext' in sys.argv:
    #     # hack hack hack
    #     for ext in torch_ffi_ext_modules:
    #         ext.build()

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
        **compile_setup_kw,
    )
