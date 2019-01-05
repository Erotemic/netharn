from os.path import exists
from os.path import dirname
from os.path import join
from os.path import sys
import re
import ast
import glob
import ubelt as ub
import os


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
    """
    require_fpath = join(dirname(__file__), fname)

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        info = {}
        if line.startswith('-e '):
            info['package'] = line.split('#egg=')[1]
        else:
            # Remove versioning from the package
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]

            info['package'] = parts[0]
            if len(parts) > 1:
                # FIXME: This breaks if the package doesnt have a version num
                op, rest = parts[1:]
                if ';' in rest:
                    # Handle platform specific dependencies
                    # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest  # NOQA
                info['version'] = (op, version)
        return info

    # This breaks on pip install, so check that it exists.
    if exists(require_fpath):
        with open(require_fpath, 'r') as f:
            packages = []
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    info = parse_line(line)
                    package = info['package']
                    if not sys.version.startswith('3.4'):
                        # apparently package_deps are broken in 3.4
                        platform_deps = info.get('platform_deps')
                        if platform_deps is not None:
                            package += ';' + platform_deps
                    packages.append(package)
            return packages
    return []


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

    abs_paths = [join(repodir, p) for pat in rel_paths for p in glob.glob(pat)]
    for abs_path in abs_paths:
        enqueue(abs_path)

    for dpath in toremove:
        # print('Removing dpath = {!r}'.format(dpath))
        ub.delete(dpath, verbose=1)


def conditional_requirements(pkgmods):
    # ONLY require these if the packages aren't installed because python
    # doesn't seem to recognize that if we already have them.
    for pkgname, modname in pkgmods:
        try:
            __import__(modname)
        except ImportError:
            yield pkgname
