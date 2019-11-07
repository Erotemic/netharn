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
from os.path import dirname
from setuptools import find_packages
from os.path import exists
from os.path import join
import ast
import glob
import ubelt as ub
import os
from skbuild import setup


def parse_version(package):
    """
    Statically parse the version number from __init__.py
    """
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
    """
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

    TODO:
        perhaps use https://github.com/davidfischer/requirements-parser instead

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        elif line.startswith('-e '):
            info = {}
            info['package'] = line.split('#egg=')[1]
            yield info
        else:
            # Remove versioning from the package
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]

            info = {}
            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    # Handle platform specific dependencies
                    # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest  # NOQA
                info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    # This breaks on pip install, so check that it exists.
    packages = []
    if exists(require_fpath):
        for info in parse_require_file(require_fpath):
            package = info['package']
            if not sys.version.startswith('3.4'):
                # apparently package_deps are broken in 3.4
                platform_deps = info.get('platform_deps')
                if platform_deps is not None:
                    package += ';' + platform_deps
            packages.append(package)
    return packages


def clean_repo(repodir, modname, rel_paths=[]):
    """
    repodir = ub.truepath('~/code/netharn/')
    modname = 'netharn'
    rel_paths = [
        'netharn/util/nms/cpu_nms.c',
        'netharn/util/nms/cpu_nms.c',
        'netharn/util/nms/cpu_nms.cpp',
        'netharn/util/nms/cython_boxes.c',
        'netharn/util/nms/cython_boxes.html',
    ]
    """
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

    import six
    if six.PY2:
        abs_paths = [join(repodir, p) for pat in rel_paths
                     for p in glob.glob(pat)]
    else:
        abs_paths = [join(repodir, p) for pat in rel_paths
                     for p in glob.glob(pat, recursive=True)]
    for abs_path in abs_paths:
        enqueue(abs_path)

    for dpath in toremove:
        # print('Removing dpath = {!r}'.format(dpath))
        ub.delete(dpath, verbose=1)


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
        'netharn/util/_nms_backend/*_nms.*so',
        'netharn/util/structs/_boxes_backend/cython_boxes.c',
        'netharn/util/structs/_boxes_backend/cython_boxes.html',
        'netharn/util/structs/_boxes_backend/cython_boxes*.*so',
        'pip-wheel-metadata',
    ]
    clean_repo(repodir, modname, rel_paths)


# Scikit-build extension module logic
compile_setup_kw = dict(
    # cmake_languages=('C', 'CXX', 'CUDA'),
    # cmake_source_dir='.',
    # cmake_args=['-DSOME_FEATURE:BOOL=OFF'],
    # cmake_source_dir='netharn',
)


version = parse_version('netharn')  # needs to be a global var for git tags

if __name__ == '__main__':
    # if 'clean' in sys.argv:
    #     # hack
    #     clean()
    #     sys.exit(0)

    setup(
        name='netharn',
        version=version,
        author='Jon Crall',
        author_email='erotemic@gmail.com',
        url='https://github.com/Erotemic/netharn',
        description='Train and deploy pytorch models',
        long_description=parse_description(),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
        },
        packages=find_packages(include='netharn.*'),
        license='Apache 2',
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
