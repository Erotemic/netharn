# -*- coding: utf-8 -*-
"""
Export component of the Pytorch exporter.

This is the code that simply exports the model toplogy via code

Uses static analysis to export relevant code that defines the model topology
into a stanadlone file. As long as your model definition is indepenent of your
training code, then the exported file can be passed around in a similar way to
a caffe prototext file.

CommandLine:
    xdoctest -m netharn.export.exporter export_model_code
    xdoctest -m netharn.export.exporter source_closure:1

    xdoctest -m netharn.export.exporter all
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ast
import re
import hashlib
import inspect
import io
import pickle
import sys
import tokenize
import types
import ubelt as ub
import warnings
from collections import OrderedDict
from os.path import abspath, join
import six

__all__ = ['export_model_code']


__pt_export_version__ = '0.4.0'


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
        print('node.__dict__ = {!r}'.format(node.__dict__))
        raise TypeError('Cannot parse a static value from non-static node '
                        'of type: {!r}'.format(type(node)))
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


class ImportVisitor(ast.NodeVisitor):
    """
    Used to search for dependencies in the original module
    """

    def __init__(visitor, fpath):
        super(ImportVisitor, visitor).__init__()
        visitor.import_names = []
        visitor.modules = []
        visitor.top_level = True
        visitor.fpath = fpath

        visitor.import_nodes = []
        visitor.import_from_nodes = []
        visitor.import_lines = {}
        visitor.assignments = {}
        pass

    def _parse_alias_list(visitor, aliases):
        for alias in aliases:
            if alias.asname is not None:
                visitor.import_names.append(alias.asname)
            else:
                if '.' not in alias.name:
                    visitor.import_names.append(alias.name)

    def visit_Import(visitor, node):
        visitor.import_nodes.append(node)
        visitor._parse_alias_list(node.names)
        visitor.generic_visit(node)

        for alias in node.names:
            key = alias.asname or alias.name
            if alias.asname:
                line = 'import {} as {}'.format(alias.name, alias.asname)
            else:
                line = 'import {}'.format(alias.name)
            visitor.import_lines[key] = line

        for alias in node.names:
            visitor.modules.append(alias.name)

    def visit_ImportFrom(visitor, node):
        visitor.import_from_nodes.append(node)
        visitor._parse_alias_list(node.names)
        visitor.generic_visit(node)

        if node.level:
            if visitor.fpath is not None:
                modparts = ub.split_modpath(abspath(visitor.fpath))[1].replace('\\', '/').split('/')
                parts = modparts[:-node.level]
                prefix = '.'.join(parts) + '.'
            else:
                prefix = '.' * node.level
        else:
            prefix = ''

        abs_modname = prefix + node.module
        visitor.modules.append(abs_modname)

        for alias in node.names:
            key = alias.asname or alias.name
            if alias.asname:
                line = 'from {} import {} as {}'.format(abs_modname, alias.name, alias.asname)
            else:
                line = 'from {} import {}'.format(abs_modname, alias.name)
            visitor.import_lines[key] = line
            # modules.append(node.level * '.' + node.module + '.' + alias.name)
            # modules.append(prefix + node.module + '.' + alias.name)

    def visit_FunctionDef(visitor, node):
        # Ignore modules imported in functions
        if not visitor.top_level:
            visitor.generic_visit(node)
            # ast.NodeVisitor.generic_visit(visitor, node)

    def visit_ClassDef(visitor, node):
        if not visitor.top_level:
            visitor.generic_visit(node)
            # ast.NodeVisitor.generic_visit(visitor, node)

    def visit_Assign(visitor, node):
        for target in node.targets:
            key = getattr(target, 'id', None)
            if key is not None:
                try:
                    value = ('static', _parse_static_node_value(node.value))
                except TypeError:
                    value = ('node', node)
                visitor.assignments[key] = value


def source_closure(model_class):
    """
    Hacky way to pull just the minimum amount of code needed to define a
    model_class.

    Args:
        model_class (type): class used to define the model_class

    Returns:
        str: closed_sourcecode: text defining a new python module.

    Example:
        >>> from torchvision import models

        >>> model_class = models.AlexNet
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> print(hash_code(text))
        18a043fc0563bcf8f97b2ee76d...

        >>> model_class = models.DenseNet
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> print(hash_code(text))
        d52175ef0d52ec5ca155bdb1037...

        >>> model_class = models.resnet50
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> print(hash_code(text))
        ad683af44142b58c85b6c2314...

        >>> model_class = models.Inception3
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> print(hash_code(text))
        bd7c67c37e292ffad6beb8532324d3...
    """
    module_name = model_class.__module__
    module = sys.modules[module_name]
    sourcecode = inspect.getsource(model_class)
    sourcecode = ub.ensure_unicode(sourcecode)
    names = undefined_names(sourcecode)

    # try:
    # module_source = ub.readfrom(module.__file__)
    # except OSError:
    module_source = inspect.getsource(module)
    module_source = ub.ensure_unicode(module_source)

    pt = ast.parse(module_source)
    visitor = ImportVisitor(module.__file__)
    try:
        visitor.visit(pt)
    except Exception:
        pass

    def closure_(obj, name):
        # TODO: handle assignments
        if name in visitor.import_lines:
            # Check and see if the name was imported from elsewhere
            return 'import', visitor.import_lines[name]
        elif name in visitor.assignments:
            type_, value = visitor.assignments[name]
            if type_ == 'node':
                # TODO, need to handle non-simple expressions
                return type_, '{} = {}'.format(name, value.value.id)
            else:
                # when value is a dict we need to be sure it is
                # extracted in the same order as we see it
                return type_, '{} = {}'.format(name, ub.repr2(value))
        elif isinstance(obj, types.FunctionType):
            if obj.__module__ == module_name:
                sourcecode = inspect.getsource(obj)
                return 'code', sourcecode
        elif isinstance(obj, type):
            if obj.__module__ == module_name:
                sourcecode = inspect.getsource(obj)
                return 'code', sourcecode

        raise NotImplementedError(str(obj) + ' ' + str(name))

    import_lines = []

    lines = [sourcecode]

    while names:
        # Make sure we process names in the same order for hashability
        names = sorted(set(names))
        for name in names:
            obj = getattr(module, name)
            type_, text = closure_(obj, name)
            if type_ == 'import':
                import_lines.append(text)
            else:
                lines.append(text)
            if text is None:
                raise NotImplementedError(str(obj) + ' ' + str(name))
                break

        import_lines = sorted(import_lines)
        closed_sourcecode = ('\n'.join(import_lines) + '\n\n\n' +
                             '\n\n'.join(lines[::-1]))
        names = sorted(undefined_names(closed_sourcecode))

    return closed_sourcecode


def remove_comments_and_docstrings(source):
    """
    Args:
        source (str): uft8 text of source code

    Returns:
        str: out: the source with comments and docstrings removed.

    References:
        https://stackoverflow.com/questions/1769332/remove-comments-docstrings

    Example:
        >>> source = ub.codeblock(
            '''
            def foo():
                'The spaces before this docstring are tokenize.INDENT'
                test = [
                    'The spaces before this string do not get a token'
                ]
            ''')
        >>> out = remove_comments_and_docstrings(source)
        >>> want = ub.codeblock(
            '''
            def foo():

                test = [
                    'The spaces before this string do not get a token'
                ]''').splitlines()
        >>> got = [o.rstrip() for o in out.splitlines()]
        >>> assert got == want

    """
    source = ub.ensure_unicode(source)
    io_obj = io.StringIO(source)
    out = ''
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (' ' * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an
                # operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out


def hash_code(sourcecode):
    r"""
    Hashes source code text, but tries to normalize things like whitespace and
    comments, so very minor changes wont change the hash.

    Args:
        source (str): uft8 text of source code

    Returns:
        str: hashid: 128 character (512 byte) hash of the normalized input

    Example:
        >>> print(hash_code('x = 1')[0:8])
        93d321be
        >>> print(hash_code('x=1 # comments and spaces dont matter')[0:8])
        93d321be
        >>> print(hash_code('\nx=1')[0:8])
        93d321be
        >>> print(hash_code('x=2')[0:8])
        6949c223
    """
    # Strip docstrings before making a parse tree
    sourcecode = ub.ensure_unicode(sourcecode)
    stripped = remove_comments_and_docstrings(sourcecode)

    # Also remove pytorch_export version info (not sure if correct?)
    stripped = re.sub('__pt_export_version__ = .*', '', stripped)

    parse_tree = ast.parse(stripped)
    # hashing the parse tree will normalize for a lot possible small changes
    ast_dump = ast.dump(parse_tree)

    hasher = hashlib.sha512()
    hasher.update(ast_dump.encode('utf8'))
    hashid = hasher.hexdigest()
    return hashid


def export_model_code(dpath, model, initkw=None):
    """
    Exports the class used to define a pytorch model as a new python module.

    Exports the minimum amount of code needed to make a self-contained Python
    module defining the pytorch model class. This exports the actual source
    code. The advantage of using this over pickle is that the original code can
    change arbitrarilly because all dependencies on the original code are
    removed in the exported code.

    Args:
        dpath (str): directory to dump the model
        model (tuple or type or object): class or class instance (e.g. torch.nn.Module)
        name (str): name to use for the file (defaults to the classname)
        initkw (dict): if specified, creates the function `make`, which
            initializes the network with the specific arguments.

    Returns:
        str: static_modpath: path to the saved model file.
            While you could put the output path in your PYTHONPATH, it is best
            to use `ub.import_module_from_path` to "load" the model instead.

    Example:
        >>> from torchvision.models import densenet
        >>> from os.path import basename
        >>> initkw = {'growth_rate': 16}
        >>> model = densenet.DenseNet(**initkw)
        >>> dpath = ub.ensure_app_cache_dir('netharn/tests')
        >>> static_modpath = export_model_code(dpath, model, initkw)
        >>> print('static_modpath = {!r}'.format(static_modpath))
        >>> print(basename(static_modpath))
        DenseNet_c662ba.py
        >>> # now the module can be loaded
        >>> module = ub.import_module_from_path(static_modpath)
        >>> loaded = module.make()
        >>> assert model.features.denseblock1.denselayer1.conv2.out_channels == 16
        >>> assert loaded.features.denseblock1.denselayer1.conv2.out_channels == 16
        >>> assert model is not loaded
    """
    if isinstance(model, type):
        model_class = model
    else:
        model_class = model.__class__
    classname = model_class.__name__

    if initkw is None:
        raise NotImplementedError(
            'ERROR: The params passed to the model __init__ must be available')
        footer = ''
    else:
        # First see if we can get away with a simple encoding of initkw
        try:
            # Do not use repr. The text produced is non-deterministic for
            # dictionaries. Instead, use ub.repr2, which is deterministic.
            init_text = ub.repr2(initkw, nl=1)
            eval(init_text, {})
            init_code = ub.codeblock(
                'initkw = {}'
            ).format(init_text)
        except Exception:
            # fallback to pickle
            warnings.warn('Initialization params might not be serialized '
                          'deterministically')
            init_bytes = repr(pickle.dumps(initkw, protocol=0))
            init_code = ub.codeblock(
                '''
                import pickle
                initkw = pickle.loads({})
                '''
            ).format(init_bytes)
        init_code = ub.indent(init_code).lstrip()
        # create a function to instanciate the class
        footer = '\n\n' + ub.codeblock(
            '''
            __pt_export_version__ = '{__pt_export_version__}'


            def get_initkw():
                """ creates an instance of the model """
                {init_code}
                return initkw


            def get_model_cls():
                model_cls = {classname}
                return model_cls


            def make():
                """ creates an instance of the model """
                initkw = get_initkw()
                model_cls = get_model_cls()
                model = model_cls(**initkw)
                return model
            '''
        ).format(classname=classname, init_code=init_code,
                 __pt_export_version__=__pt_export_version__)

        # TODO: assert that the name "make" is not used in the model body

    body = source_closure(model_class)

    body_footer = body + footer + '\n'
    # dont need to hash the header, because comments are removed anyway
    hashid = hash_code(body_footer)

    header = ub.codeblock(
        '''
        """
        This module was autogenerated by netharn/export/exporter.py
        original_module={}
        classname={}
        timestamp={}
        hashid={}
        """
        ''').format(model_class.__module__, classname, ub.timestamp(), hashid)

    sourcecode = header + '\n' + body_footer

    static_modname = classname + '_' + hashid[0:6]
    static_modpath = join(dpath, static_modname + '.py')
    with open(static_modpath, 'w') as file:
        file.write(sourcecode)
    return static_modpath
