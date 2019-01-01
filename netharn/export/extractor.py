"""
Extracts relevant parts of the source code
"""
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


def source_closure(model_class, expand_modules=[]):
    """
    Hacky way to pull just the minimum amount of code needed to define a
    model_class. Uses a combination of dynamic and static introspection.

    Args:
        model_class (type):
            class used to define the model_class

        expand_modules (List[str]):
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
        >>> import netharn as nh
        >>> model_class = nh.layers.ConvNormNd
        >>> expand_modules = ['netharn']
        >>> text = source_closure(model_class, expand_modules)
        >>> print(text)
    """
    module_name = model_class.__module__
    module = sys.modules[module_name]

    # Extract the source code of the class only
    class_sourcecode = inspect.getsource(model_class)
    class_sourcecode = ub.ensure_unicode(class_sourcecode)

    # Initialize a list to accumulate code until everything is closed
    import_lines = []
    lines = [class_sourcecode]

    # Parse the parent module to find only the relevant global varaibles and
    # include those in the extracted source code.
    visitor = ImportVisitor.parse_module(module)
    visitor.import_info

    while True:
        # Determine if there are any variables needed from the parent scope
        current_header = '\n'.join(sorted(import_lines))
        current_body = '\n\n'.join(lines[::-1])
        current_sourcecode = (current_header + '\n\n\n' + current_body)
        names = sorted(undefined_names(current_sourcecode))

        if not names:
            # Exit the loop once all variables are defined
            break

        # Make sure we process names in the same order for hashability
        names = sorted(set(names))
        for name in names:
            try:
                type_, text = visitor.closure(name)
            except NotImplementedError:
                current_header = '\n'.join(sorted(import_lines))
                current_body = '\n\n'.join(lines[::-1])
                print('--- <ERROR> ---')
                print('Error computing source code closure')
                print(' * failed to close name = {!r}'.format(name))
                print('<<< CURRENT_HEADER >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_header)))
                print('<<< CURRENT_BODY >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_body)))
                print('--- </ERROR> ---')
                raise
            if type_ == 'import':
                import_lines.append(text)
            else:
                lines.append(text)
            if text is None:
                raise NotImplementedError(str(name))
                break

    if expand_modules:
        # Expand references to internal modules
        closed_visitor = ImportVisitor.parse_source(current_sourcecode)
        self = closed_visitor.import_info
        for root in expand_modules:
            needs_expansion = self.root_to_varnames.get(root, [])
            for varname in needs_expansion:
                info = self.varname_to_info[varname]
                modpath = ub.modname_to_modpath(info['modname'])

                if info['type'] == 'attribute':
                    # We can directly replace this import statement by
                    # copy-pasting the relevant code from the other module
                    # (ASSUMING THERE ARE NO NAME CONFLICTS)

                    # TODO: Now we just need to get the closure with respect to
                    # these newly added variables.
                    new = ImportVisitor.parse_source(open(modpath, 'r').read(), modpath)
                    a = new.closure(varname)
                    print('TODO: NEED TO CLOSE attribute varname = {!r}'.format(varname))
                    # print('a = {!r}'.format(a))

                elif info['type'] == 'module':
                    # TODO: We need to determine what actually is used from
                    print('TODO: NEED TO CLOSE module varname = {!r}'.format(varname))
                    # this module and fix some of the names.
                    pass

        # for root in closed_visitor.import_info.root_modnames:
        #     if root in expand_modules:
        #         print('TOEXPAND root = {!r}'.format(root))

    closed_sourcecode = current_sourcecode
    return closed_sourcecode


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
    def __init__(self, fpath=None):
        self.fpath = fpath

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
            if self.fpath is not None:
                try:
                    rel_modpath = ub.split_modpath(abspath(self.fpath))[1]
                except ValueError:
                    warnings.warn('fpath={} does not exist'.format(self.fpath))
                    rel_modpath = basename(abspath(self.fpath))
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


class ImportVisitor(ast.NodeVisitor):
    """
    Used to search for dependencies in the original module

    References:
        https://greentreesnakes.readthedocs.io/en/latest/nodes.html

    Example:
        >>> from netharn.export.extractor import *
        >>> from netharn.export import extractor
        >>> fpath = extractor.__file__
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
        >>> visitor = ImportVisitor.parse_source(sourcecode, fpath=fpath)
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

    def __init__(visitor, fpath=None, module_name=None, module=None):
        super(ImportVisitor, visitor).__init__()
        visitor.fpath = fpath
        visitor.module_name = module_name
        visitor.module = module

        visitor.import_info = ImportInfo(fpath=fpath)
        visitor.assignments = {}

        visitor.calldefs = {}
        visitor.top_level = True

    def finalize(visitor):
        visitor.import_info.finalize()

    @classmethod
    def parse_source(ImportVisitor, module_source, fpath=None):
        pt = ast.parse(module_source)
        visitor = ImportVisitor(fpath=fpath)
        visitor.visit(pt)
        visitor.finalize()
        return visitor

    @classmethod
    def parse_module(ImportVisitor, module):
        module_source = inspect.getsource(module)
        module_source = ub.ensure_unicode(module_source)
        pt = ast.parse(module_source)
        visitor = ImportVisitor(module.__file__, module.__name__, module)
        visitor.visit(pt)
        visitor.finalize()
        return visitor

    def closure(visitor, name):
        """
        Given a live-object and its assigned name in a file find the lines of
        code that define it.
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
                raise AssertionError('Need module to dynamic analysis')
            obj = getattr(visitor.module, name)
            if isinstance(obj, types.FunctionType):
                if obj.__module__ == visitor.module_name:
                    sourcecode = inspect.getsource(obj)
                    return 'code', sourcecode
            elif isinstance(obj, type):
                if obj.__module__ == visitor.module_name:
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
