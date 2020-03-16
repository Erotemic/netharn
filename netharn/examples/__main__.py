

def main():
    import xdoctest
    import ubelt as ub
    import sys

    modpath = ub.modname_to_modpath('netharn.examples')
    name_to_path = {}
    submods = list(xdoctest.static_analysis.package_modpaths(modpath))
    for submod in submods:
        modname = ub.augpath(submod, dpath='', ext='')
        if not modname.startswith('_'):
            name_to_path[modname] = submod

    print('name_to_path = {}'.format(ub.repr2(name_to_path, nl=1)))

    chosen = None
    for arg in sys.argv[1:2]:
        print('arg = {!r}'.format(arg))
        if arg in name_to_path:
            chosen = name_to_path[arg]
            break
    print('chosen = {!r}'.format(chosen))

    assert chosen is not None
    module = ub.import_module_from_path(chosen)
    print('module = {!r}'.format(module))
    module.main()

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.examples
    """
    main()
