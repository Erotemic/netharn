from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import range, cStringIO
import six
import operator
import atexit
import sys
import re
import itertools as it
from collections import defaultdict
import parse

if '--profile' in sys.argv:
    import line_profiler
    profile = line_profiler.LineProfiler()
    IS_PROFILING = True
else:
    def __dummy_profile__(func):
        """ dummy profiling func. does nothing """
        return func
    profile = __dummy_profile__
    IS_PROFILING = False

if '--flamegraph' in sys.argv:
    """
    https://github.com/evanhempel/python-flamegraph
    python -m flamegraph -o perf.log myscript.py --your-script args here
    """
    import flamegraph
    flamegraph.start_profile_thread(fd=open("./perf.log", "w"))


@atexit.register
def _dump_global_profile_report():
    # if we are profiling, then dump out info at the end of the program
    if IS_PROFILING:
        self = _KernprofParser(profile)
        self.dump_text()


def profile_onthefly(func):
    """
    Automatically dumps a profile report whenever decorated function is called

    Example:
        >>> def foo():
        >>>     a = [i for i in range(10)]
        >>>     b = [i for i in range(100)]
        >>>     c = [i for i in range(1000)]
        >>> profile_onthefly(foo)()
    """
    import line_profiler
    profile = line_profiler.LineProfiler()
    new_func = profile(func)
    new_func.profile_info = _KernprofParser(profile)
    new_func.print_report = new_func.profile_info.print_report

    def wraper(*args, **kwargs):
        retval = new_func(*args, **kwargs)
        new_func.print_report()
        return retval
    wraper.new_func = new_func
    return wraper


class _KernprofParser(object):

    def __init__(self, profile):
        self.profile = profile

    def raw_text(self):
        file_ = cStringIO()
        self.profile.print_stats(stream=file_, stripzeros=True)
        file_.seek(0)
        text =  file_.read()
        return text

    def print_report(self):
        print(self.raw_text())

    def get_text(self):
        text = self.raw_text()
        output_text, summary_text = self.clean_line_profile_text(text)
        return output_text, summary_text

    def dump_text(self):
        import ubelt as ub
        print("Dumping Profile Information")
        try:
            output_text, summary_text = self.get_text()
        except AttributeError:
            print('profile is not on')
        else:
            #profile.dump_stats('out.lprof')
            print(summary_text)
            suffix = ub.argval('--profname', default='')
            if suffix:
                suffix = '_' + suffix
            ub.writeto('profile_output{}.txt'.format(suffix), output_text + '\n' + summary_text)
            ub.writeto('profile_output{}.{}.txt'.format(suffix, ub.timestamp()),
                       output_text + '\n' + summary_text)

    def parse_rawprofile_blocks(self, text):
        """
        Split the file into blocks along delimters and and put delimeters back in
        the list
        """
        # The total time reported in the raw output is from pystone not kernprof
        # The pystone total time is actually the average time spent in the function
        delim = 'Total time: '
        delim2 = 'Pystone time: '
        #delim = 'File: '
        profile_block_list = re.split('^' + delim, text, flags=re.MULTILINE | re.DOTALL)
        for ix in range(1, len(profile_block_list)):
            profile_block_list[ix] = delim2 + profile_block_list[ix]
        return profile_block_list

    def clean_line_profile_text(self, text):
        """
        Sorts the output from line profile by execution time
        Removes entries which were not run
        """
        #
        profile_block_list = self.parse_rawprofile_blocks(text)
        #profile_block_list = fix_rawprofile_blocks(profile_block_list)
        #---
        # FIXME can be written much nicer
        prefix_list, timemap = self.parse_timemap_from_blocks(profile_block_list)
        # Sort the blocks by time
        sorted_lists = sorted(six.iteritems(timemap), key=operator.itemgetter(0))
        newlist = prefix_list[:]
        for key, val in sorted_lists:
            newlist.extend(val)
        # Rejoin output text
        output_text = '\n'.join(newlist)
        #---
        # Hack in a profile summary
        summary_text = self.get_summary(profile_block_list)
        output_text = output_text
        return output_text, summary_text

    def get_block_totaltime(self, block):

        def get_match_text(match):
            if match is not None:
                start, stop = match.start(), match.end()
                return match.string[start:stop]
            else:
                return None

        time_line = get_match_text(re.search('Pystone time: [0-9.]* s', block, flags=re.MULTILINE | re.DOTALL))
        if time_line is None:
            time_str = None
        else:
            time_str = get_match_text(re.search('[0-9.]+', time_line, flags=re.MULTILINE | re.DOTALL))
        if time_str is not None:
            return float(time_str)
        else:
            return None

    def get_block_id(self, block, readlines=None):

        def named_field(key, regex, vim=False):
            return r'(?P<%s>%s)' % (key, regex)

        non_unicode_whitespace = '[^ \t\n\r\f\v]'

        fpath_regex = named_field('fpath', non_unicode_whitespace + '+')
        funcname_regex = named_field('funcname', non_unicode_whitespace + '+')
        lineno_regex = named_field('lineno', '[0-9]+')

        fileline_regex = 'File: ' + fpath_regex + '$'
        funcline_regex = 'Function: ' + funcname_regex + ' at line ' + lineno_regex + '$'
        fileline_match = re.search(fileline_regex, block, flags=re.MULTILINE)
        funcline_match = re.search(funcline_regex, block, flags=re.MULTILINE)
        if fileline_match is not None and funcline_match is not None:
            fpath    = fileline_match.groupdict()['fpath']
            funcname = funcline_match.groupdict()['funcname']
            lineno   = funcline_match.groupdict()['lineno']
            # TODO: Determine if the function belongs to a class

            if readlines:
                # TODO: make robust
                classname = _find_parent_class(fpath, funcname, lineno, readlines)
                if classname:
                    funcname = classname + '.' + funcname
                # try:
                #     lines = readlines(fpath)
                #     row = int(lineno) - 1
                #     funcline = lines[row]
                #     if not funcline.startswith('def'):
                #         # get indentation
                #         indent = len(funcline) - len(funcline).lstrip()
                #         # function is nested. fixme
                #         funcname = '<nested>:' + funcname
                #         _find_pyclass_above_row(lines, row, indent)
                # except Exception:
                #     funcname = '<inspect_error>:' + funcname
            block_id = funcname + ':' + fpath + ':' + lineno
        else:
            block_id = 'None:None:None'
        return block_id

    def parse_timemap_from_blocks(self, profile_block_list):
        """
        Build a map from times to line_profile blocks
        """
        prefix_list = []
        timemap = defaultdict(list)
        for ix in range(len(profile_block_list)):
            block = profile_block_list[ix]
            total_time = self.get_block_totaltime(block)
            # Blocks without time go at the front of sorted output
            if total_time is None:
                prefix_list.append(block)
            # Blocks that are not run are not appended to output
            elif total_time != 0:
                timemap[total_time].append(block)
        return prefix_list, timemap

    def get_summary(self, profile_block_list, maxlines=20):
        """
        References:
            https://github.com/rkern/line_profiler
        """
        import ubelt as ub
        time_list = [self.get_block_totaltime(block) for block in profile_block_list]
        time_list = [time if time is not None else -1 for time in time_list]

        @ub.memoize
        def readlines(fpath):
            return open(fpath, 'r').readlines()

        blockid_list = [self.get_block_id(block, readlines=readlines)
                        for block in profile_block_list]
        sortx = ub.argsort(time_list)
        sorted_time_list = list(ub.take(time_list, sortx))
        sorted_blockid_list = list(ub.take(blockid_list, sortx))

        aligned_blockid_list = _align_lines(sorted_blockid_list, ':')
        summary_lines = [('%6.2f seconds - ' % time) + line
                         for time, line in
                         zip(sorted_time_list, aligned_blockid_list)]

        summary_text = '\n'.join(summary_lines[-maxlines:])
        return summary_text

    def fix_rawprofile_blocks(self, profile_block_list):
        # TODO: finish function. should multiply times by
        # Timer unit to get true second profiling
        #profile_block_list_new = []
        for block in profile_block_list:
            block_lines = block.split('\n')
            sep = ['=' * 62]
            def split_block_at_sep(block_lines, sep):
                for pos, line in enumerate(block_lines):
                    if line.find(sep) == 0:
                        pos += 1
                        header_lines = block_lines[:pos]
                        body_lines = block_lines[pos:]
                        return header_lines, body_lines
                return block_lines, None
            header_lines, body_lines = split_block_at_sep(block_lines, sep)

    def clean_lprof_file(self, input_fname, output_fname=None):
        """ Reads a .lprof file and cleans it """
        # Read the raw .lprof text dump
        text = open(input_fname, 'r').read()
        # Sort and clean the text
        output_text = self.clean_line_profile_text(text)
        return output_text


def _find_parent_class(fpath, funcname, lineno, readlines=None):
    """
    Example:
        >>> from netharn.util import profiler
        >>> import ubelt as ub
        >>> funcname = 'clean_lprof_file'
        >>> func = getattr(profiler._KernprofParser, funcname)
        >>> lineno = func.__code__.co_firstlineno
        >>> fpath = profiler.__file__
        >>> #fpath = ub.truepath('~/code/netharn/netharn/util/profiler.py')
        >>> #lineno   = 264
        >>> readlines = lambda x: open(x, 'r').readlines()
        >>> classname = _find_parent_class(fpath, funcname, lineno, readlines)
        >>> print('classname = {!r}'.format(classname))
        >>> assert classname == '_KernprofParser'
    """
    if readlines is None:
        def readlines(fpath):
            return open(fpath, 'r').readlines()

    try:
        line_list = readlines(fpath)
        row = int(lineno) - 1
        funcline = line_list[row]
        indent = len(funcline) - len(funcline.lstrip())
        if indent > 0:
            # get indentation
            # function is nested. fixme
            funcname = '<nested>:' + funcname
            return _find_pyclass_above_row(line_list, row, indent)
    except Exception:
        pass


def _find_pyclass_above_row(line_list, row, indent):
    """
    originally part of the vim plugin

    HACK: determine the class of the profiled funcs
    """
    # Get text posision
    pattern = '^class [a-zA-Z_]'
    classline, classpos = _find_pattern_above_row(pattern, line_list, row,
                                                  indent, maxIter=None)
    result = parse.parse('class {name}({rest}', classline)
    classname = result.named['name']
    return classname


def _find_pattern_above_row(pattern, line_list, row, indent, maxIter=None):
    """
    searches a few lines above the curror until it **matches** a pattern
    """
    # Iterate until we match.
    # Janky way to find function / class name
    retval = None

    for ix in it.count(0):
        pos = row - ix
        if maxIter is not None and ix > maxIter:
            break
        if pos < 0:
            break
        searchline = line_list[pos]

        if indent is not None:
            if not searchline.strip():
                continue
            search_n_indent = len(searchline) - len(searchline.lstrip())
            if indent <= search_n_indent:
                continue
            # if indent < search_n_indent:
            #     continue

        if re.match(pattern, searchline) is not None:
            retval = searchline, pos
            break
    return retval


def _align_lines(line_list, character='=', replchar=None, pos=0):
    r"""
    Left justifies text on the left side of character

    TODO:
        clean up and move to ubelt?

    Args:
        line_list (list of strs):
        character (str):
        pos (int or list or None): does one alignment for all chars beyond this
            column position. If pos is None, then all chars are aligned.

    Returns:
        list: new_lines

    Example:
        >>> line_list = 'a = b\none = two\nthree = fish'.split('\n')
        >>> character = '='
        >>> new_lines = _align_lines(line_list, character)
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        a     = b
        one   = two
        three = fish

    Example:
        >>> line_list = 'foofish:\n    a = b\n    one    = two\n    three    = fish'.split('\n')
        >>> character = '='
        >>> new_lines = _align_lines(line_list, character)
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        foofish:
            a        = b
            one      = two
            three    = fish

    Example:
        >>> import ubelt as ub
        >>> character = ':'
        >>> text = ub.codeblock('''
            {'max': '1970/01/01 02:30:13',
             'mean': '1970/01/01 01:10:15',
             'min': '1970/01/01 00:01:41',
             'range': '2:28:32',
             'std': '1:13:57',}''').split('\n')
        >>> new_lines = _align_lines(text, ':', ' :')
        >>> result = '\n'.join(new_lines)
        >>> print(result)
        {'max'   : '1970/01/01 02:30:13',
         'mean'  : '1970/01/01 01:10:15',
         'min'   : '1970/01/01 00:01:41',
         'range' : '2:28:32',
         'std'   : '1:13:57',}

    Example:
        >>> line_list = 'foofish:\n a = b = c\n one = two = three\nthree=4= fish'.split('\n')
        >>> character = '='
        >>> # align the second occurence of a character
        >>> new_lines = _align_lines(line_list, character, pos=None)
        >>> print(('\n'.join(line_list)))
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        foofish:
         a   = b   = c
         one = two = three
        three=4    = fish
    """

    # FIXME: continue to fix ansi
    if pos is None:
        # Align all occurences
        num_pos = max([line.count(character) for line in line_list])
        pos = list(range(num_pos))

    # Allow multiple alignments
    if isinstance(pos, list):
        pos_list = pos
        # recursive calls
        new_lines = line_list
        for pos in pos_list:
            new_lines = _align_lines(new_lines, character=character,
                                     replchar=replchar, pos=pos)
        return new_lines

    # base case
    if replchar is None:
        replchar = character

    # the pos-th character to align
    lpos = pos
    rpos = lpos + 1

    tup_list = [line.split(character) for line in line_list]

    handle_ansi = True
    if handle_ansi:
        # Remove ansi from length calculation
        # References: http://stackoverflow.com/questions/14693701remove-ansi
        ansi_escape = re.compile(r'\x1b[^m]*m')

    # Find how much padding is needed
    maxlen = 0
    for tup in tup_list:
        if len(tup) >= rpos + 1:
            if handle_ansi:
                tup = [ansi_escape.sub('', x) for x in tup]
            left_lenlist = list(map(len, tup[0:rpos]))
            left_len = sum(left_lenlist) + lpos * len(replchar)
            maxlen = max(maxlen, left_len)

    # Pad each line to align the pos-th occurence of the chosen character
    new_lines = []
    for tup in tup_list:
        if len(tup) >= rpos + 1:
            lhs = character.join(tup[0:rpos])
            rhs = character.join(tup[rpos:])
            # pad the new line with requested justification
            newline = lhs.ljust(maxlen) + replchar + rhs
            new_lines.append(newline)
        else:
            new_lines.append(replchar.join(tup))
    return new_lines


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.profiler all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
