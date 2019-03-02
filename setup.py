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
     # TODO: linuxmany distribution https://www.python.org/dev/peps/pep-0513/

     # In the meantime, build a pure python version without binary files
     # not sure exactly how to do this so just hack it.
     python setup.py bdist_wheel --universal
     find . -iname *.so -d

     # Use twine to upload. This will prompt for username and password
     twine upload --username erotemic --skip-existing dist/*

     # Check the url to make sure everything worked
     https://pypi.org/project/netharn/

     # ---------- OLD ----------------
     # Check the url to make sure everything worked
     https://pypi.python.org/pypi?:action=display&name=netharn

"""
from __future__ import absolute_import, division, print_function
import sys
import os
from os.path import dirname
from setuptools import find_packages

import skbuild_pr
import skbuild
skbuild.conditional_requirements = skbuild_pr.conditional_requirements
skbuild.clean_repo = skbuild_pr.clean_repo
skbuild.parse_description = skbuild_pr.parse_description
skbuild.parse_requirements = skbuild_pr.parse_requirements
skbuild.parse_version = skbuild_pr.parse_version

COMPILE_MODE = os.environ.get('COMPILE_MODE', 'skbuild')
if COMPILE_MODE == 'skbuild':
    from skbuild import setup  # This line replaces 'from setuptools import setup'
elif COMPILE_MODE == 'pure_python':
    from setuptools import setup  # NOQA
else:
    raise KeyError(COMPILE_MODE)


def clean():
    """
    __file__ = ub.truepath('~/code/netharn/setup.py')
    """
    modname = 'netharn'
    repodir = dirname(__file__)
    rel_paths = [
        'htmlcov',
        '_skbuild',
        '_build_wheel',
        'netharn.egg-info',
        'dist',
        'build',
        '**/*.pyc',
        'profile*'
        'netharn/util/_nms_backend/cpu_nms.c',
        'netharn/util/_nms_backend/cpu_nms.c',
        'netharn/util/_nms_backend/cpu_nms.cpp',
        'netharn/util/_boxes_backend/cython_boxes.c',
        'netharn/util/_boxes_backend//cython_boxes.html',
        'netharn/util/_nms_backend/*_nms.*so',
        'netharn/util/_boxes_backend/cython_boxes*.*so'
    ]
    skbuild.clean_repo(repodir, modname, rel_paths)


# Scikit-build extension module logic
if COMPILE_MODE == 'skbuild':

    # print(sys.argv)

    print('sys.argv = {!r}'.format(sys.argv))

    compile_setup_kw = dict(
        cmake_source_dir='.',
        # cmake_args=['-DSOME_FEATURE:BOOL=OFF'],
        # cmake_source_dir='netharn',
    )
    if '-DUSE_CUDA:BOOL=FALSE' in sys.argv:
        # EXTREME HACK: not sure if this is even necessary
        compile_setup_kw['cmake_languages'] = ('C', 'CXX')
    else:
        compile_setup_kw['cmake_languages'] = ('C', 'CXX', 'CUDA')
else:
    compile_setup_kw = {}


version = skbuild.parse_version('netharn')  # needs to be a global var for git tags

if __name__ == '__main__':
    if 'clean' in sys.argv:
        # hack
        clean()
        sys.exit(0)

    setup(
        name='netharn',
        version=version,
        author='Jon Crall',
        description='Train and deploy pytorch models',
        long_description=skbuild.parse_description(),
        install_requires=skbuild.parse_requirements('requirements.txt'),
        extras_require={
            'all': skbuild.parse_requirements('optional-requirements.txt')
        },
        author_email='erotemic@gmail.com',
        url='https://github.com/Erotemic/netharn',
        license='Apache 2',
        packages=find_packages(include='netharn.*'),
        # packages=['netharn',],
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Utilities',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3.6',
        ],
        **compile_setup_kw
    )
