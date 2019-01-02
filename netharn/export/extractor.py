"""
Extracts relevant parts of the source code
"""
from os.path import isdir
from os.path import join
from os.path import basename
from collections import OrderedDict
import warnings
import ast
import astunparse
import inspect
import six
import types
import ubelt as ub
from os.path import abspath
from os.path import sys
# TODO Fix issue with from . import statements

# __all__ = [
#     'source_closure',
# ]

DEBUG = 0


def source_closure(model_class, expand_names=[]):
    """
    Hacky way to pull just the minimum amount of code needed to define a
    model_class. Uses a combination of dynamic and static introspection.

    Args:
        model_class (type):
            class used to define the model_class

        expand_names (List[str]):
            EXPERIMENTAL. List of modules that should be expanded into raw
            source code.

    Returns:
        str: closed_sourcecode: text defining a new python module.

    CommandLine:
        xdoctest -m netharn.export.extractor source_closure

    Example:
        >>> import torchvision
        >>> from torchvision import models
        >>> got = {}

        >>> model_class = models.AlexNet
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['alexnet'] = ub.hash_data(text)

        >>> model_class = models.DenseNet
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['densenet'] = ub.hash_data(text)

        >>> model_class = models.resnet50
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['resnet50'] = ub.hash_data(text)

        >>> model_class = models.Inception3
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['inception3'] = ub.hash_data(text)

        >>> # Thvisitor.import_infoe hashes will depend on torchvision itself
        >>> if torchvision.__version__ == '0.2.1':
        >>>     want = {
        >>>         'alexnet': '8342d8ef40c7898ab84191',
        >>>         'densenet': 'b32246b3437a2321d5ffb',
        >>>         'resnet50': '4182eeffbee94b003d556',
        >>>         'inception3': '8888d9e4b97593c84cd',
        >>>     }
        >>>     failed = []
        >>>     for k in want:
        >>>         if not got[k].startswith(want[k]):
        >>>             item = (k, got[k], want[k])
        >>>             print('failed item = {!r}'.format(item))
        >>>             failed.append(item)
        >>>     assert not failed, str(failed)
        >>> else:
        >>>     warnings.warn('Unsupported version of torchvision')

    Doctest:
        >>> # Test a heavier duty class
        >>> from netharn.export.extractor import *
        >>> import netharn as nh
        >>> model_class = nh.device.MountedModel
        >>> model_class = nh.layers.ConvNormNd
        >>> model_class = nh.layers.Sequential
        >>> from netharn.models.yolo2.light_yolo import Yolo
        >>> model_class = Yolo
        >>> expand_names = ['netharn', 'ubelt']
        >>> expand_names = ['netharn']
        >>> from ovharn.models import multiscale_mcd_arch
        >>> model_class = multiscale_mcd_arch.Multiscale_MCD_Resnet50
        >>> expand_names = ['ovharn']
        >>> text = source_closure(model_class, expand_names)
        >>> from netharn.export.exporter import remove_comments_and_docstrings
        >>> text = remove_comments_and_docstrings(text)
        >>> print(text)
    """
    closer = Closer()
    closer.add_dynamic(model_class)
    if expand_names:
        closer.expand(expand_names)
    closed_sourcecode = closer.current_sourcecode()
    return closed_sourcecode


class Closer(object):
    """
    Maintains the current state of the source code extract_definition
    """
    def __init__(closer):
        closer.body_lines = []
        closer.header_lines = []

    def current_sourcecode(closer):
        current_header = '\n'.join(sorted(closer.header_lines))
        current_body = '\n\n'.join(closer.body_lines[::-1])
        current_sourcecode = (current_header + '\n\n\n' + current_body)
        return current_sourcecode

    def add_dynamic(closer, obj):
        """
        Add the source to define a live python object
        """
        modname = obj.__module__
        module = sys.modules[modname]

        # Extract the source code of the class only
        class_sourcecode = inspect.getsource(obj)
        class_sourcecode = ub.ensure_unicode(class_sourcecode)
        closer.body_lines.append(class_sourcecode)
        visitor = ImportVisitor.parse(module=module)
        closer.close(visitor)

    def add_static(closer, name, modpath):
        visitor = ImportVisitor.parse(modpath=modpath)
        closer._extract_and_populate(visitor, name)
        closer.close(visitor)

    def _extract_and_populate(closer, visitor, name):
        try:
            type_, text = visitor.extract_definition(name)
        except NotImplementedError:
            current_sourcecode = closer.current_sourcecode()
            print('--- <ERROR> ---')
            print('Error computing source code extract_definition')
            print(' * failed to close name = {!r}'.format(name))
            print('<<< CURRENT_SOURCE >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_sourcecode)))
            print('--- </ERROR> ---')
            raise
        if text is None:
            if DEBUG:
                warnings.warn(str(name))
                return
            else:
                raise NotImplementedError(str(name))
        if type_ == 'import':
            closer.header_lines.append(text)
        else:
            closer.body_lines.append(text)

    def close(closer, visitor):
        """
        Populate all undefined names using the context from a module
        """
        # Parse the parent module to find only the relevant global varaibles and
        # include those in the extracted source code.
        current_sourcecode = closer.current_sourcecode()

        # Loop until all undefined names are defined
        names = True
        while names:
            # Determine if there are any variables needed from the parent scope
            current_sourcecode = closer.current_sourcecode()
            # Make sure we process names in the same order for hashability
            prev_names = names
            names = sorted(undefined_names(current_sourcecode))
            if names == prev_names:
                if DEBUG:
                    warnings.warn('We were unable do do anything about undefined names')
                    return
                else:
                    raise AssertionError('unable to define names')
            for name in names:
                closer._extract_and_populate(visitor, name)

    def replace_varname(closer, find, repl):
        repl_header = [line if line != find else '# ' + line
                       for line in repl.header_lines]

        for i, line in enumerate(closer.header_lines):
            if line == find:
                closer.header_lines[i] = '# ' + line

        closer.header_lines.extend(repl_header)
        closer.body_lines.extend(repl.body_lines)
        closer.header_lines = list(ub.unique(closer.header_lines))

    def expand_module_attributes(closer, varname):
        current_sourcecode = closer.current_sourcecode()
        closed_visitor = ImportVisitor.parse(source=current_sourcecode)
        info = closed_visitor.import_info.varname_to_info[varname]
        find = info['line']
        varmodpath = ub.modname_to_modpath(info['modname'])
        expansion = info['expansion']

        def _exhaust(name, modname, modpath):
            current_sourcecode = closer.current_sourcecode()
            closed_visitor = ImportVisitor.parse(source=current_sourcecode)
            print('REWRITE ACCESSOR name = {!r}'.format(name))
            rewriter = RewriteModuleAccess(name)
            rewriter.visit(closed_visitor.pt)
            new_body = astunparse.unparse(closed_visitor.pt)
            closer.body_lines = [new_body]

            for subname in rewriter.accessed_attrs:
                submodname = modname + '.' + subname
                submodpath = ub.modname_to_modpath(submodname)
                if submodpath is not None:
                    # if the accessor is to another module, exhaust until
                    # we reach a non-module
                    print('EXAUSTING: {}, {}, {}'.format(subname, submodname, submodpath))
                    _exhaust(subname, submodname, submodpath)
                else:
                    # Otherwise we can directly add the referenced attribute
                    print('FINALIZE: {}'.format(subname, modpath))
                    closer.add_static(subname, modpath)

        _exhaust(varname, expansion, varmodpath)
        new_closer = Closer()
        repl = new_closer
        closer.replace_varname(find, repl)

    def expand(closer, expand_names):
        """
        Experimental feature
        """
        print("!!! EXPANDING")
        # Expand references to internal modules
        flag = True
        while flag:
            flag = False
            current_sourcecode = closer.current_sourcecode()
            closed_visitor = ImportVisitor.parse(source=current_sourcecode)
            self = closed_visitor.import_info  # NOQA
            for root in expand_names:
                needs_expansion = closed_visitor.import_info.root_to_varnames.get(root, [])
                for varname in needs_expansion:
                    flag = True
                    info = closed_visitor.import_info.varname_to_info[varname]
                    modpath = ub.modname_to_modpath(info['modname'])

                    if info['type'] == 'attribute':
                        # We can directly replace this import statement by
                        # copy-pasting the relevant code from the other module
                        # (ASSUMING THERE ARE NO NAME CONFLICTS)

                        # TODO: Now we just need to get the extract_definition with respect to
                        # these newly added variables.
                        find = closed_visitor.import_info.varname_to_info[varname]['line']
                        print('TODO: NEED TO CLOSE attribute varname = {!r}'.format(varname))
                        print(' * modpath = {!r}'.format(modpath))
                        print(' * find = {!r}'.format(find))
                        new_closer = Closer()
                        name = varname
                        new_closer.add_static(name, modpath)
                        new_closer.expand(expand_names)

                        # Replace the import with the definition
                        repl = new_closer
                        closer.replace_varname(find, repl)
                        print('CLOSED attribute varname = {}'.format(varname))
                        # a = new.extract_definition(varname)
                        # print('a = {!r}'.format(a))

                    elif info['type'] == 'module':
                        print('TODO: NEED TO CLOSE module varname = {!r}'.format(varname))
                        closer.expand_module_attributes(varname)
                        print('CLOSED module varname = {}'.format(varname))
                        current_sourcecode = closer.current_sourcecode()
                        closed_visitor = ImportVisitor.parse(source=current_sourcecode)

                        # # TODO: We need to determine what actually is used from
                        # find = closed_visitor.import_info.varname_to_info[varname]['line']

                        # # FIXME: Rewrite only the appropriate section
                        # rewriter = RewriteModuleAccess(varname)
                        # rewriter.visit(closed_visitor.pt)
                        # new_body = astunparse.unparse(closed_visitor.pt)
                        # closer.body_lines = [new_body]

                        # expansion = info['expansion']
                        # print(' * modpath = {!r}'.format(modpath))
                        # print(' * expansion = {!r}'.format(expansion))
                        # print(' * rewriter.accessed_attrs = {!r}'.format(rewriter.accessed_attrs))
                        # print(' * find = {!r}'.format(find))
                        # new_closer = Closer()
                        # accessed_submodules = []

                        # for name in rewriter.accessed_attrs:
                        #     x = expansion + '.' + name
                        #     if ub.modname_to_modpath(x):
                        #         accessed_submodules.append((x, name))
                        #     else:
                        #         closer.add_static(name, modpath)

                        # for submod, name in accessed_submodules:
                        #     # should probably have a recursive component
                        #     print('submod = {!r}'.format(submod))
                        #     print('name = {!r}'.format(name))
                        #     rewriter = RewriteModuleAccess(name)
                        #     rewriter.visit(closed_visitor.pt)
                        #     new_body = astunparse.unparse(closed_visitor.pt)
                        #     closer.body_lines = [new_body]

                        #     new_submod = []
                        #     print('rewriter.accessed_attrs = {!r}'.format(rewriter.accessed_attrs))
                        #     submodpath = ub.modname_to_modpath(submod)
                        #     print('submodpath = {!r}'.format(submodpath))

                        #     for name in rewriter.accessed_attrs:
                        #         x = submod + '.' + name
                        #         if ub.modname_to_modpath(x):
                        #             new_submod.append((x, name))
                        #         else:
                        #             print('SS name = {!r}'.format(name))
                        #             print('submodpath = {!r}'.format(submodpath))
                        #             closer.add_static(name, submodpath)

                        # new_closer.expand(expand_names)

                        # FIXME: Rewrite only the appropriate section
                        # current_sourcecode = closer.current_sourcecode()
                        # closed_visitor = ImportVisitor.parse(source=current_sourcecode)
                        # rewriter = RewriteModuleAccess(varname)
                        # rewriter.visit(closed_visitor.pt)
                        # new_body = astunparse.unparse(closed_visitor.pt)
                        # closer.body_lines = [new_body]

                        # new_closer = Closer()
                        # repl = new_closer
                        # new_closer.expand(expand_names)
                        # closer.replace_varname(find, repl)

                        # TODO:
                        #     - [ ] Determine which attributes of "varname" are accessed

                        # this module and fix some of the names.
                        pass


def _parse_static_node_value(node):
    """
    Extract a constant value from a node if possible
    """
    if isinstance(node, ast.Num):
        value = node.n
    elif isinstance(node, ast.Str):
        value = node.s
    elif isinstance(node, ast.List):
        value = list(map(_parse_static_node_value, node.elts))
    elif isinstance(node, ast.Tuple):
        value = tuple(map(_parse_static_node_value, node.elts))
    elif isinstance(node, (ast.Dict)):
        keys = map(_parse_static_node_value, node.keys)
        values = map(_parse_static_node_value, node.values)
        value = OrderedDict(zip(keys, values))
        # value = dict(zip(keys, values))
    elif six.PY3 and isinstance(node, (ast.NameConstant)):
        value = node.value
    elif (six.PY2 and isinstance(node, ast.Name) and
          node.id in ['None', 'True', 'False']):
        # disregard pathological python2 corner cases
        value = {'None': None, 'True': True, 'False': False}[node.id]
    else:
        msg = ('Cannot parse a static value from non-static node '
               'of type: {!r}'.format(type(node)))
        # print('node.__dict__ = {!r}'.format(node.__dict__))
        # print('msg = {!r}'.format(msg))
        raise TypeError(msg)
    return value


def undefined_names(sourcecode):
    """
    Parses source code for undefined names

    Example:
        >>> print(ub.repr2(undefined_names('x = y'), nl=0))
        {'y'}
    """
    import pyflakes.api
    import pyflakes.reporter

    class CaptureReporter(pyflakes.reporter.Reporter):
        def __init__(reporter, warningStream, errorStream):
            reporter.syntax_errors = []
            reporter.messages = []
            reporter.unexpected = []

        def unexpectedError(reporter, filename, msg):
            reporter.unexpected.append(msg)

        def syntaxError(reporter, filename, msg, lineno, offset, text):
            reporter.syntax_errors.append(msg)

        def flake(reporter, message):
            reporter.messages.append(message)

    names = set()

    reporter = CaptureReporter(None, None)
    pyflakes.api.check(sourcecode, '_.py', reporter)
    for msg in reporter.messages:
        if msg.__class__.__name__.endswith('UndefinedName'):
            assert len(msg.message_args) == 1
            names.add(msg.message_args[0])
    return names


class ImportInfo(ub.NiceRepr):
    """
    Hold information about module-level imports for ImportVisitor
    """
    def __init__(self, modpath=None):
        self.modpath = modpath

        self.varnames = []

        self.varname_to_expansion = {}
        self.varname_to_line = {}
        self.root_to_varnames = ub.ddict(list)

        self._import_nodes = []
        self._import_from_nodes = []

    def finalize(self):
        self.root_to_varnames = ub.ddict(list)
        self.varname_to_info = {}

        for varname, expansion in self.varname_to_expansion.items():

            info = {
                'varname': varname,
                'expansion': expansion,
                'line': self.varname_to_line[varname],
            }
            longest_modname = None
            parts = expansion.split('.')
            for i in range(1, len(parts) + 1):
                root = '.'.join(parts[:i])
                if ub.modname_to_modpath(root):
                    longest_modname = root
                self.varname_to_info
                self.root_to_varnames[root].append(varname)

            info['modname'] = longest_modname
            if info['expansion'] == info['modname']:
                info['type'] = 'module'
            else:
                info['type'] = 'attribute'
            self.varname_to_info[varname] = info

        if 0:
            print(ub.repr2(self.varname_to_info))

    def __nice__(self):
        return ub.repr2(self.varname_to_line, nl=1)

    def register_import_node(self, node):
        self._import_nodes.append(node)

        for alias in node.names:
            varname = alias.asname or alias.name
            if alias.asname:
                line = 'import {} as {}'.format(alias.name, alias.asname)
            else:
                line = 'import {}'.format(alias.name)
            self.varname_to_line[varname] = line
            self.varname_to_expansion[varname] = alias.name

    def register_from_import_node(self, node):
        self._import_from_nodes.append(node)

        if node.level:
            # Handle relative imports
            if self.modpath is not None:
                try:
                    rel_modpath = ub.split_modpath(abspath(self.modpath))[1]
                except ValueError:
                    warnings.warn('modpath={} does not exist'.format(self.modpath))
                    rel_modpath = basename(abspath(self.modpath))
                modparts = rel_modpath.replace('\\', '/').split('/')
                parts = modparts[:-node.level]
                prefix = '.'.join(parts)
                if node.module:
                    prefix = prefix + '.'
            else:
                warnings.warn('Unable to rectify absolute import')
                prefix = '.' * node.level
        else:
            prefix = ''

        if node.module is not None:
            abs_modname = prefix + node.module
        else:
            abs_modname = prefix

        for alias in node.names:
            varname = alias.asname or alias.name
            if alias.asname:
                line = 'from {} import {} as {}'.format(abs_modname, alias.name, alias.asname)
            else:
                line = 'from {} import {}'.format(abs_modname, alias.name)
            self.varname_to_line[varname] = line
            self.varname_to_expansion[varname] = abs_modname + '.' + alias.name


class RewriteModuleAccess(ast.NodeTransformer):
    """
    Example:
        >>> from netharn.export.extractor import *
        >>> source = ub.codeblock(
        ...     '''
        ...     foo.bar = 3
        ...     foo.baz.bar = 3
        ...     biz.foo.baz.bar = 3
        ...     ''')
        >>> pt = ast.parse(source)
        >>> visitor = RewriteModuleAccess('foo')
        >>> orig = astunparse.unparse(pt)
        >>> print(orig)
        >>> visitor.visit(pt)
        >>> modified = astunparse.unparse(pt)
        >>> print(modified)
        >>> visitor.accessed_attrs

    """
    def __init__(self, modname):
        self.modname = modname
        self.level = 0
        self.accessed_attrs = []

    def visit_Import(self, node):
        if self.level == 0:
            return None
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Str):
            return None
        return node

    def visit_ImportFrom(self, node):
        if self.level == 0:
            return None
        return node

    def visit_FunctionDef(self, node):
        self.level += 1
        self.generic_visit(node)
        self.level -= 1
        return node

    def visit_ClassDef(self, node):
        self.level += 1
        self.generic_visit(node)
        self.level -= 1
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            if node.value.id == self.modname:
                self.accessed_attrs.append(node.attr)
                new_node = ast.Name(node.attr, node.ctx)
                old_node = node
                return ast.copy_location(new_node, old_node)
        return node


class ImportVisitor(ast.NodeVisitor):
    """
    Used to search for dependencies in the original module

    References:
        https://greentreesnakes.readthedocs.io/en/latest/nodes.html

    Example:
        >>> from netharn.export.extractor import *
        >>> from netharn.export import extractor
        >>> modpath = extractor.__file__
        >>> sourcecode = ub.codeblock(
        ...     '''
        ...     import a
        ...     import b
        ...     import c.d
        ...     import e.f as g
        ...     from . import h
        ...     from .i import j
        ...     from . import k, l, m
        ...     from n import o, p, q
        ...     ''')
        >>> visitor = ImportVisitor.parse_source(sourcecode, modpath=modpath)
        >>> print(ub.repr2(visitor.import_info.varname_to_info))
        ...
        >>> print(visitor.import_info)
        <ImportInfo({
            'a': 'import a',
            'b': 'import b',
            'c.d': 'import c.d',
            'g': 'import e.f as g',
            'h': 'from netharn.export import h',
            'j': 'from netharn.export.i import j',
            'k': 'from netharn.export import k',
            'l': 'from netharn.export import l',
            'm': 'from netharn.export import m',
            'o': 'from n import o',
            'p': 'from n import p',
            'q': 'from n import q',
        })>
    """

    def __init__(visitor, modpath=None, modname=None, module=None):
        super(ImportVisitor, visitor).__init__()
        visitor.pt = None
        visitor.modpath = modpath
        visitor.modname = modname
        visitor.module = module

        visitor.import_info = ImportInfo(modpath=modpath)
        visitor.assignments = {}

        visitor.calldefs = {}
        visitor.top_level = True

    def finalize(visitor):
        visitor.import_info.finalize()

    @classmethod
    def parse(ImportVisitor, source=None, modpath=None, modname=None,
              module=None):
        if module is not None:
            source = inspect.getsource(module)
            modname = module.__name__
            modpath = module.__file__

        if modpath is not None:
            if isdir(modpath):
                modpath = join(modpath, '__init__.py')
            if modname is None:
                modname = ub.modpath_to_modname(modpath)

        if modpath is not None:
            if source is None:
                source = open(modpath, 'r').read()

        if source is None:
            raise ValueError('unable to derive source code')

        source = ub.ensure_unicode(source)
        pt = ast.parse(source)
        visitor = ImportVisitor(modpath, modname, module)
        visitor.pt = pt
        visitor.visit(pt)
        visitor.finalize()
        return visitor

    def extract_definition(visitor, name):
        """
        Given the name of a variable / class / function / moodule, extract the
        relevant lines of source code that define that structure from the
        visited module.
        """
        # if name == 'fcn_coder':
        #     return 'import', 'from ovharn.models import fcn_coder'
        if name in visitor.import_info.varname_to_line:
            # Check and see if the name was imported from elsewhere
            return 'import', visitor.import_info.varname_to_line[name]
        elif name in visitor.assignments:
            # TODO: better handling of assignments
            type_, value = visitor.assignments[name]
            if type_ == 'node':
                # Use ast unparser to generate the rhs of the assignment
                # May be able to use astunparse elsewhere to reduce bloat
                value = astunparse.unparse(value.value)  # .rstrip()
                return type_, '{} = {}'.format(name, value)
            elif type_ == 'static':
                # We were able to pre-extract a static value.
                # Note, when value is a dict we need to be sure it is extracted
                # in the same order as we see it
                return type_, '{} = {}'.format(name, ub.repr2(value))
            else:
                raise NotImplementedError(type_)
        elif name in visitor.calldefs:
            sourcecode = astunparse.unparse(visitor.calldefs[name])
            return 'code', sourcecode
        else:
            # Fallback to dynamic analysis
            # NOTE: now that we are tracking calldefs and using astunparse,
            # this code should not need to be called.
            if visitor.module is None:
                print('visitor = {!r}'.format(visitor))
                print('visitor.modpath = {!r}'.format(visitor.modpath))
                print('visitor.import_info = {!r}'.format(visitor.import_info))
                if DEBUG:
                    warnings.warn('Need module to dynamic analysis: {}'.format(name))
                    return None, None
                else:
                    raise AssertionError('Need module to dynamic analysis: {}'.format(name))
            obj = getattr(visitor.module, name)
            if isinstance(obj, types.FunctionType):
                if obj.__module__ == visitor.modname:
                    sourcecode = inspect.getsource(obj)
                    return 'code', sourcecode
            elif isinstance(obj, type):
                if obj.__module__ == visitor.modname:
                    sourcecode = inspect.getsource(obj)
                    return 'code', sourcecode
            raise NotImplementedError(str(obj) + ' ' + str(name))

    def visit_Import(visitor, node):
        visitor.import_info.register_import_node(node)
        visitor.generic_visit(node)

    def visit_ImportFrom(visitor, node):
        visitor.import_info.register_from_import_node(node)
        visitor.generic_visit(node)

    def visit_Assign(visitor, node):
        for target in node.targets:
            key = getattr(target, 'id', None)
            if key is not None:
                try:
                    value = ('static', _parse_static_node_value(node.value))
                except TypeError:
                    value = ('node', node)
                visitor.assignments[key] = value

    def visit_FunctionDef(visitor, node):
        visitor.calldefs[node.name] = node
        # Ignore any non-top-level imports
        if not visitor.top_level:
            visitor.generic_visit(node)
            # ast.NodeVisitor.generic_visit(visitor, node)

    def visit_ClassDef(visitor, node):
        visitor.calldefs[node.name] = node
        # Ignore any non-top-level imports
        if not visitor.top_level:
            visitor.generic_visit(node)
            # ast.NodeVisitor.generic_visit(visitor, node)
