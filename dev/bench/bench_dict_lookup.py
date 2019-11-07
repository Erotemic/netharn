"""
How much do different ways of accessing variables hurt us in critical loops?
"""


def _gen_cluttered_func(n=100):
    lines = []
    import ubelt as ub
    import kwarray
    rng = kwarray.ensure_rng(0)

    varnames = []
    for i in range(n):
        mode = rng.choice(['int', 'float', 'str'])
        if mode == 'int':
            value = rng.randint(0, 100000)
        if mode == 'str':
            value = ub.hash_data(rng.randint(0, 100000))[0:10]
        if mode == 'float':
            value = rng.randn() * 1000
        varname = 'var{:03d}'.format(i)
        line = '{} = {!r}'.format(varname, value)
        lines.append(line)
        varnames.append(varname)

    clutter_vars = ub.indent('\n'.join(lines))

    template = ub.codeblock(
        '''
        def {FUNCNAME}():
        {CLUTTER}
            ignore_inf_loss_parts = d['ignore_inf_loss_parts']
            for i in range(num_inner_loops):
                if ignore_inf_loss_parts:
                    pass
            # return {RETVAL}
        ''')

    retval = '[{}]'.format(','.join(varnames))
    funcname = 'clutter_{}'.format(n)

    text = template.format(FUNCNAME=funcname, CLUTTER=clutter_vars, RETVAL=retval)
    return text, funcname


def main():
    import ubelt as ub
    header = ub.codeblock(
        '''
        import ubelt as ub
        ti = ub.Timerit(100, bestof=10, verbose=2)

        d = {
            'keyboard_debug': False,
            'snapshot_after_error': True,  # Try to checkpoint before crashing
            'show_prog': True,
            'use_tqdm': None,
            'prog_backend': 'progiter',
            'ignore_inf_loss_parts': False,
            'use_tensorboard': True,
            'export_modules': [],
            'large_loss': 1000,
            'num_keep': 2,
            'keep_freq': 20,
        }

        num_inner_loops = 10000

        def access_dict_direct():
            for i in range(num_inner_loops):
                if d['ignore_inf_loss_parts']:
                    pass
        for timer in ti.reset('access_dict_direct'):
            with timer:
                access_dict_direct()

        ''')

    parts = [header]

    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]:
        func_text, funcname = _gen_cluttered_func(n=n)
        time_text = ub.codeblock(
            '''
            {func_text}
            for timer in ti.reset('{funcname}'):
                with timer:
                    {funcname}()
            ''').format(func_text=func_text, funcname=funcname)
        parts.append(time_text)

    block = '\n'.join(parts)

    prog_text = ub.codeblock(
        '''
        import ubelt as ub
        def main():
        {block}

        if __name__ == '__main__':
            main()
        ''').format(block=ub.indent(block))

    # prog_text = 'def main():\n' + ub.indent(block) + 'if __name__ == "__main__":\n    main()'
    fpath = 'bench_local_clutter.py'
    with open(fpath, 'w') as file:
        file.write(prog_text)

    ub.cmd('python ' + fpath, verbose=3)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/bench/bench_dict_lookup.py
    """
    main()
