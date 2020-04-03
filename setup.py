#!/usr/bin/env python
# -*- coding: utf-8 -*-
# NOTE: pip install -U --pre h5py
from __future__ import absolute_import, division, print_function
import sys
from os.path import dirname
from setuptools import find_packages
from os.path import exists
from os.path import join
import glob
import os
from setuptools import setup
# from skbuild import setup


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        pandoc --from=markdown --to=rst --output=README.rst README.md
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.rst')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        try:
            with open(readme_fpath, 'r') as f:
                text = f.read()
            return text
        except Exception as ex:
            import warnings
            warnings.warn('unable to parse existing readme: {!r}'.format(ex))
    return ''


def parse_requirements(fname='requirements.txt', with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items

    References:
        https://pip.readthedocs.io/en/1.1/requirements.html

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(setup.parse_requirements(with_version=True)))"
    """
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line, base='.'):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith(('-f ', '--find-links ', '--index-url ')):
            import warnings
            warnings.warn(
                'requirements file specified alternative index urls, but '
                'there is currently no way to support this in setuptools')
        elif line.startswith('-r '):
            # Allow specifying requirements in other files
            new_fname = line.split(' ')[1]
            new_fpath = join(base, new_fname)
            for info in parse_require_file(new_fpath):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

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
        base = dirname(fpath)
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line, base):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def clean_repo(repodir, modname, rel_paths=[]):
    """
    repodir = ub.expandpath('~/code/netharn/')
    modname = 'netharn'
    rel_paths = [
        'netharn/util/nms/cpu_nms.c',
        'netharn/util/nms/cpu_nms.c',
        'netharn/util/nms/cpu_nms.cpp',
        'netharn/util/nms/cython_boxes.c',
        'netharn/util/nms/cython_boxes.html',
    ]
    """
    print('cleaning repo: {}/{}'.format(repodir, modname))
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
            if fname.endswith('.pyc'):
                toremove.append(join(root, fname))
            if fname.endswith(('.so', '.c', '.o')):
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

    import ubelt as ub
    for dpath in toremove:
        # print('Removing dpath = {!r}'.format(dpath))
        ub.delete(dpath, verbose=1)


def clean():
    """
    __file__ = ub.expandpath('~/code/netharn/setup.py')
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
        'pip-wheel-metadata',
    ]
    clean_repo(repodir, modname, rel_paths)


VERSION = version = parse_version('netharn/__init__.py')  # needs to be a global var for git tags
NAME = 'netharn'

if __name__ == '__main__':
    if 'clean' in sys.argv:
        clean()
        # sys.exit(0)

    setup(
        name=NAME,
        version=VERSION,
        author='Jon Crall',
        author_email='jon.crall@kitware.com',
        url='https://gitlab.kitware.com/computer-vision/netharn',
        description='Train and deploy pytorch models',
        long_description=parse_description(),
        long_description_content_type='text/x-rst',
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
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Utilities',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3',
        ],
    )
