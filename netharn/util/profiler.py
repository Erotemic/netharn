from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import range, cStringIO
import six
import operator
import atexit
import sys
import re
import itertools as it
from collections import defaultdict

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
def dump_global_profile_report():
    # if we are profiling, then dump out info at the end of the program
    if IS_PROFILING:
        self = KernprofParser(profile)
        # print('----')
        # print('RAW')
        # print('----')
        # self.print_report()
        # print('----')
        # print('DUMPING')
        # print('----')
        self.dump_text()


def dynamic_profile(func):
    import line_profiler
    profile = line_profiler.LineProfiler()
    new_func = profile(func)
    new_func.profile_info = KernprofParser(profile)
    new_func.print_report = new_func.profile_info.print_report
    return new_func


def profile_onthefly(func):
    import line_profiler
    profile = line_profiler.LineProfiler()
    new_func = profile(func)
    new_func.profile_info = KernprofParser(profile)
    new_func.print_report = new_func.profile_info.print_report

    def wraper(*args, **kwargs):
        retval = new_func(*args, **kwargs)
        new_func.print_report()
        return retval
    wraper.new_func = new_func
    return wraper


class KernprofParser(object):

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
                classname = find_parent_class(fpath, funcname, lineno, readlines)
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
                #         find_pyclass_above_row(lines, row, indent)
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

        import utool as ut
        aligned_blockid_list = ut.util_str.align_lines(sorted_blockid_list, ':')
        summary_lines = [('%6.2f seconds - ' % time) + line
                         for time, line in
                         zip(sorted_time_list, aligned_blockid_list)]
        #summary_header = ut.codeblock(
        #    '''
        #    CLEANED PROFILE OUPUT

        #    The Pystone timings are not from kernprof, so they may include kernprof
        #    overhead, whereas kernprof timings do not (unless the line being
        #    profiled is also decorated with kernrof)

        #    The kernprof times are reported in Timer Units

        #    ''')
        # summary_lines_ = ut.listclip(summary_lines, maxlines, fromback=True)
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


def find_parent_class(fpath, funcname, lineno, readlines=None):
    """
    Example:
        >>> from netharn.util import profiler
        >>> import ubelt as ub
        >>> funcname = 'clean_lprof_file'
        >>> func = getattr(profiler.KernprofParser, funcname)
        >>> lineno = func.__code__.co_firstlineno
        >>> fpath = profiler.__file__
        >>> #fpath = ub.truepath('~/code/netharn/netharn/util/profiler.py')
        >>> #lineno   = 264
        >>> readlines = lambda x: open(x, 'r').readlines()
        >>> classname = find_parent_class(fpath, funcname, lineno, readlines)
        >>> print('classname = {!r}'.format(classname))
        >>> assert classname == 'KernprofParser'
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
            return find_pyclass_above_row(line_list, row, indent)
    except Exception:
        pass


def find_pyclass_above_row(line_list, row, indent):
    """
    originally part of the vim plugin

    HACK: determine the class of the profiled funcs
    """
    # Get text posision
    pattern = '^class [a-zA-Z_]'
    classline, classpos = find_pattern_above_row(pattern, line_list, row,
                                                 indent, maxIter=None)
    import parse
    result = parse.parse('class {name}({rest}', classline)
    classname = result.named['name']
    return classname


def find_pattern_above_row(pattern, line_list, row, indent, maxIter=None):
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


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.profiler all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
