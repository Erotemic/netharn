"""
Simple script that prints the deployed models in a given netharn work directory
"""
import scriptconfig as scfg
import ubelt as ub
import glob
from os.path import join, exists


class ListDeployedConfig(scfg.Config):
    """
    Given a netharn work directory list all deployed models
    """
    default = {
        'workdir': scfg.Value(None, help='work directory'),
        'name': scfg.Value(None, help='"nice" name of the run'),
    }


def main(cmdline=True, **kw):
    config = ListDeployedConfig(cmdline=cmdline, default=kw)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    runs_dpath = join(config['workdir'], 'fit/runs')
    if not exists(runs_dpath):
        print('Workdir does not seem to contain a runs dpath')
        print('Checking for alternates? TODO')
        raise NotImplementedError

    workdirs = [config['workdir']]
    for workdir in workdirs:
        run_name = config['name']
        if run_name is None:
            named_run_dpath = join(runs_dpath, '*')
            dpath_exists = exists(named_run_dpath)
            print('dpath_exists = {!r}'.format(dpath_exists))
        else:
            named_run_dpath = join(runs_dpath, run_name)
            dpath_exists = exists(named_run_dpath)
            print('dpath_exists = {!r}'.format(dpath_exists))

        # TODO: do we want to remove deploy.zip symlinks here?
        deployed_fpaths = glob.glob(join(named_run_dpath, '*/*.zip'))

        SHRINK = 1
        if SHRINK:
            # Make output text smaller, and more likely to work cross-system
            deployed_fpaths = [
                ub.shrinkuser(fpath, home='$HOME')
                for fpath in deployed_fpaths]

        print('deployed_fpaths = {}'.format(ub.repr2(deployed_fpaths, nl=1)))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/list_deployed.py --workdir $HOME/work/netharn
    """
    main()
