#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
# -*- coding: utf-8 -*-
"""
Requirements:
    pip install gitpython click ubelt
"""
import re
from os.path import exists
from os.path import join
from os.path import dirname
from os.path import abspath
try:
    import ubelt as ub
    import click
    import git as gitpython
    import functools
except Exception as ex:

    ALLOW_FALLBACK = True
    if ALLOW_FALLBACK:
        print('Attempting to install requirements/super_setup.txt')
        import subprocess
        import sys
        try:
            super_setup_dpath = dirname(__file__)
        except NameError:
            super_setup_dpath = '.'  # For Ipython (assume in repo root)
        superreq_fpath = join(super_setup_dpath, 'requirements/super_setup.txt')
        args = [sys.executable, '-m', 'pip', 'install', '-r', superreq_fpath]
        proc = subprocess.Popen(args)
        out, err = proc.communicate()
        print(out)
        print(err)

    try:
        import ubelt as ub
        import click
        import git as gitpython
        import functools
    except Exception:
        FALLBACK_FAILED = True
    else:
        FALLBACK_FAILED = False

    if FALLBACK_FAILED:
        print('ex = {!r}'.format(ex))
        print('!!!!!!!!!')
        print('NEED TO INSTALL SUPER SETUP DEPENDENCIES. RUN:')
        print('pip install -r requirements/super_setup.txt')
        print('!!!!!!!!!')
        raise


class ShellException(Exception):
    """
    Raised when shell returns a non-zero error code
    """


class DirtyRepoError(Exception):
    """
    If the repo is in an unexpected state, its very easy to break things using
    automated scripts. To be safe, we don't do anything. We ensure this by
    raising this error.
    """


def parse_version(package):
    """
    Statically parse the version number from __init__.py

    CommandLine:
        python -c "import setup; print(setup.parse_version('ovharn'))"
    """
    from os.path import dirname, join
    import ast
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


class GitURL(ub.NiceRepr):
    """
    Represent and transform git urls between protocols defined in [3]_.

    The code in GitURL is largely derived from [1]_ and [2]_.
    Credit to @coala and @FriendCode.

    Note:
        while this code aims to suport protocols defined in [3]_, it is only
        tested for specific use cases and therefore might need to be improved.

    References:
        .. [1] https://github.com/coala/git-url-parse
        .. [2] https://github.com/FriendCode/giturlparse.py
        .. [3] https://git-scm.com/docs/git-clone#URLS

    Example:
        >>> # xdoctest: +SKIP
        >>> self = GitURL('git@gitlab.kitware.com:computer-vision/netharn.git')
        >>> print(ub.repr2(self.parts()))
        >>> print(self.format('ssh'))
        >>> print(self.format('https'))
        >>> self = GitURL('https://gitlab.kitware.com/computer-vision/netharn.git')
        >>> print(ub.repr2(self.parts()))
        >>> print(self.format('ssh'))
        >>> print(self.format('https'))
    """
    SYNTAX_PATTERNS = {
        # git allows for a url style syntax
        'url': re.compile(r'(?P<transport>\w+://)'
                          r'((?P<user>\w+[^@]*@))?'
                          r'(?P<host>[a-z0-9_.-]+)'
                          r'((?P<port>:[0-9]+))?'
                          r'/(?P<path>.*\.git)'),
        # git allows for ssh style syntax
        'ssh': re.compile(r'(?P<user>\w+[^@]*@)'
                          r'(?P<host>[a-z0-9_.-]+)'
                          r':(?P<path>.*\.git)'),
    }

    r"""
    Ignore:
        # Helper to build the parse pattern regexes
        def named(key, regex):
            return '(?P<{}>{})'.format(key, regex)

        def optional(pat):
            return '({})?'.format(pat)

        parse_patterns = {}
        # Standard url format
        transport = named('transport', r'\w+://')
        user = named('user', r'\w+[^@]*@')
        host = named('host', r'[a-z0-9_.-]+')
        port = named('port', r':[0-9]+')
        path = named('path', r'.*\.git')

        pat = ''.join([transport, optional(user), host, optional(port), '/', path])
        parse_patterns['url'] = pat

        pat = ''.join([user, host, ':', path])
        parse_patterns['ssh'] = pat
        print(ub.repr2(parse_patterns))
    """

    def __init__(self, url):
        self._url = url
        self._parts = None

    def __nice__(self):
        return self._url

    def parts(self):
        """
        Parses a GIT URL and returns an info dict.

        Returns:
            dict: info about the url

        Raises:
            Exception : if parsing fails
        """
        info = {
            'syntax': '',
            'host': '',
            'user': '',
            'port': '',
            'path': None,
            'transport': '',
        }

        for syntax, regex in self.SYNTAX_PATTERNS.items():
            match = regex.search(self._url)
            if match:
                info['syntax'] = syntax
                info.update(match.groupdict())
                break
        else:
            raise Exception('Invalid URL {!r}'.format(self._url))

        # change none to empty string
        for k, v in info.items():
            if v is None:
                info[k] = ''
        return info

    def format(self, protocol):
        """
        Change the protocol of the git URL
        """
        parts = self.parts()
        if protocol == 'ssh':
            parts['user'] = 'git@'
            url = ''.join([
                parts['user'], parts['host'], ':', parts['path']
            ])
        else:
            parts['transport'] = protocol + '://'
            parts['port'] = ''
            parts['user'] = ''
            url = ''.join([
                parts['transport'], parts['user'], parts['host'],
                parts['port'], '/', parts['path']
            ])
        return url


class Repo(ub.NiceRepr):
    """
    Abstraction that references a git repository, and is able to manipulate it.

    A common use case is to define a `remote` and a `code_dpath`, which lets
    you check and ensure that the repo is cloned and on a particular branch.
    You can also query its status, and pull, and perform custom git commands.

    Args:
        *args: name, dpath, code_dpath, remotes, remote, branch

    Attributes:
        All names listed in args are attributse. In addition, the class also
        exposes these derived attributes.

        url (URI): where the primary location is

    Example:
        >>> # xdoctest: +SKIP
        >>> # Here is a simple example referencing ubelt
        >>> from super_setup import *
        >>> import ubelt as ub
        >>> repo = Repo(
        >>>     remote='https://github.com/Erotemic/ubelt.git',
        >>>     code_dpath=ub.ensuredir(ub.expandpath('~/tmp/demo-repos')),
        >>> )
        >>> print('repo = {}'.format(repo))
        >>> repo.check()
        >>> repo.ensure()
        >>> repo.check()
        >>> repo.status()
        >>> repo._cmd('python setup.py build')
        >>> repo._cmd('./run_doctests.sh')
        repo = <Repo('ubelt')>

        >>> # Here is a less simple example referencing ubelt
        >>> from super_setup import *
        >>> import ubelt as ub
        >>> repo = Repo(
        >>>     name='ubelt-local',
        >>>     remote='github',
        >>>     branch='master',
        >>>     remotes={
        >>>         'github': 'https://github.com/Erotemic/ubelt.git',
        >>>         'fakemirror': 'https://gitlab.com/Erotemic/ubelt.git',
        >>>     },
        >>>     code_dpath=ub.ensuredir(ub.expandpath('~/tmp/demo-repos')),
        >>> )
        >>> print('repo = {}'.format(repo))
        >>> repo.ensure()
        >>> repo._cmd('python setup.py build')
        >>> repo._cmd('./run_doctests.sh')
    """
    def __init__(repo, **kwargs):
        repo.name = kwargs.pop('name', None)
        repo.dpath = kwargs.pop('dpath', None)
        repo.code_dpath = kwargs.pop('code_dpath', None)
        repo.remotes = kwargs.pop('remotes', None)
        repo.remote = kwargs.pop('remote', None)
        repo.branch = kwargs.pop('branch', 'master')

        repo._logged_lines = []
        repo._logged_cmds = []

        if repo.remote is None:
            if repo.remotes is None:
                raise ValueError('must specify some remote')
            else:
                if len(repo.remotes) > 1:
                    raise ValueError('remotes are ambiguous, specify one')
                else:
                    repo.remote = ub.peek(repo.remotes)
        else:
            if repo.remotes is None:
                _default_remote = 'origin'
                repo.remotes = {
                    _default_remote: repo.remote
                }
                repo.remote = _default_remote

        repo.url = repo.remotes[repo.remote]

        if repo.name is None:
            suffix = repo.url.split('/')[-1]
            repo.name = suffix.split('.git')[0]

        if repo.dpath is None:
            repo.dpath = join(repo.code_dpath, repo.name)

        repo.pkg_dpath = join(repo.dpath, repo.name)

        for path_attr in ['dpath', 'code_dpath']:
            path = getattr(repo, path_attr)
            if path is not None:
                setattr(repo, path_attr, ub.expandpath(path))

        repo.verbose = kwargs.pop('verbose', 3)
        if kwargs:
            raise ValueError('unknown kwargs = {}'.format(kwargs.keys()))

        repo._pygit = None

    def set_protocol(self, protocol):
        """
        Changes the url protocol to either ssh or https

        Args:
            protocol (str): can be ssh or https
        """
        # Update base url to use the requested protocol
        gurl = GitURL(self.url)
        self.url = gurl.format(protocol)
        # Update all remote urls to use the requested protocol
        for key in list(self.remotes.keys()):
            self.remotes[key] = GitURL(self.remotes[key]).format(protocol)

    def info(repo, msg):
        repo._logged_lines.append(('INFO', 'INFO: ' + msg))
        if repo.verbose >= 1:
            print(msg)

    def debug(repo, msg):
        repo._logged_lines.append(('DEBUG', 'DEBUG: ' + msg))
        if repo.verbose >= 1:
            print(msg)

    def _getlogs(repo):
        return '\n'.join([t[1] for t in repo._logged_lines])

    def __nice__(repo):
        return '{}, branch={}'.format(repo.name, repo.branch)

    def _cmd(repo, command, cwd=ub.NoParam, verbose=ub.NoParam):
        if verbose is ub.NoParam:
            verbose = repo.verbose
        if cwd is ub.NoParam:
            cwd = repo.dpath

        repo._logged_cmds.append((command, cwd))
        repo.debug('Run {!r} in {!r}'.format(command, cwd))

        info = ub.cmd(command, cwd=cwd, verbose=verbose)

        if verbose:
            if info['out'].strip():
                repo.info(info['out'])

            if info['err'].strip():
                repo.debug(info['err'])

        if info['ret'] != 0:
            raise ShellException(ub.repr2(info))
        return info

    @property
    # @ub.memoize_property
    def pygit(repo):
        """ pip install gitpython """
        if repo._pygit is None:
            repo._pygit = gitpython.Repo(repo.dpath)
        return repo._pygit

    def develop(repo):
        if ub.WIN32:
            # We can't run a shell file on win32, so lets hope this works
            import warnings
            warnings.warn('super_setup develop may not work on win32')
            repo._cmd('pip install -e .', cwd=repo.dpath)
        else:
            devsetup_script_fpath = join(repo.dpath, 'run_developer_setup.sh')
            if not exists(devsetup_script_fpath):
                raise AssertionError('Assume we always have run_developer_setup.sh: repo={!r}'.format(repo))
            repo._cmd(devsetup_script_fpath, cwd=repo.dpath)

    def doctest(repo):
        if ub.WIN32:
            raise NotImplementedError('doctest does not yet work on windows')

        devsetup_script_fpath = join(repo.dpath, 'run_doctests.sh')
        if not exists(devsetup_script_fpath):
            raise AssertionError('Assume we always have run_doctests.sh: repo={!r}'.format(repo))
        repo._cmd(devsetup_script_fpath, cwd=repo.dpath)

    def clone(repo):
        if exists(repo.dpath):
            raise ValueError('cannot clone into non-empty directory')
        args = '--recursive'
        # NOTE: if the remote branch does not exist this will fail
        if repo.branch is not None:
            args += ' -b {}'.format(repo.branch)
        try:
            command = 'git clone {args} {url} {dpath}'.format(
                args=args, url=repo.url, dpath=repo.dpath)
            repo._cmd(command, cwd=repo.code_dpath)
        except Exception as ex:
            text = repr(ex)
            if 'Remote branch' in text and 'not found' in text:
                print('ERROR: It looks like the remote branch you asked for doesnt exist')
                print('ERROR: Caused by: ex = {}'.format(text))
                raise Exception('Cannot find branch {} for repo {}'.format(repo.branch, repo))
            raise

    def _assert_clean(repo):
        if repo.pygit.is_dirty():
            raise DirtyRepoError('The repo={} is dirty'.format(repo))

    def check(repo):
        repo.ensure(dry=True)

    def versions(repo):
        """
        Print current version information
        """
        fmtkw = {}
        fmtkw['pkg'] = parse_version(repo.pkg_dpath) + ','
        fmtkw['sha1'] = repo._cmd('git rev-parse HEAD', verbose=0)['out'].strip()
        try:
            fmtkw['tag'] = repo._cmd('git describe --tags', verbose=0)['out'].strip() + ','
        except ShellException:
            fmtkw['tag'] = '<None>,'
        fmtkw['branch'] = repo.pygit.active_branch.name + ','
        fmtkw['repo'] = repo.name + ','
        repo.info('repo={repo:<14} pkg={pkg:<12} tag={tag:<18} branch={branch:<10} sha1={sha1}'.format(
            **fmtkw))

    def ensure_clone(repo):
        if exists(repo.dpath):
            repo.debug('No need to clone existing repo={}'.format(repo))
        else:
            repo.debug('Clone non-existing repo={}'.format(repo))
            repo.clone()

    def ensure(repo, dry=False):
        """
        Ensure that the repo is checked out on your local machine, that the
        correct branch is checked out, and the upstreams are targeting the
        correct remotes.
        """
        if repo.verbose > 0:
            if dry:
                repo.debug(ub.color_text('Checking {}'.format(repo), 'blue'))
            else:
                repo.debug(ub.color_text('Ensuring {}'.format(repo), 'blue'))

        if not exists(repo.dpath):
            repo.debug('NEED TO CLONE {}: {}'.format(repo, repo.url))
            if dry:
                return

        repo.ensure_clone()

        repo._assert_clean()

        # Ensure all registered remotes exist
        for remote_name, remote_url in repo.remotes.items():
            try:
                remote = repo.pygit.remotes[remote_name]
                have_urls = list(remote.urls)
                if remote_url not in have_urls:
                    # TODO supress this warning if its just a git vs https
                    # thing using GitURL
                    print('WARNING: REMOTE NAME EXISTS BUT URL IS NOT {}. '
                          'INSTEAD GOT: {}'.format(remote_url, have_urls))
            except (IndexError):
                try:
                    print('NEED TO ADD REMOTE {}->{} FOR {}'.format(
                        remote_name, remote_url, repo))
                    if not dry:
                        repo._cmd('git remote add {} {}'.format(remote_name, remote_url))
                except ShellException:
                    if remote_name == repo.remote:
                        # Only error if the main remote is not available
                        raise

        # Ensure we have the right remote
        try:
            remote = repo.pygit.remotes[repo.remote]
        except IndexError:
            if not dry:
                raise AssertionError('Something went wrong')
            else:
                remote = None

        if remote is not None:
            try:
                if not remote.exists():
                    raise IndexError
                else:
                    repo.debug('The requested remote={} name exists'.format(remote))
            except IndexError:
                repo.debug('WARNING: remote={} does not exist'.format(remote))
            else:
                if remote.exists():
                    repo.debug('Requested remote does exists')
                    remote_branchnames = [ref.remote_head for ref in remote.refs]
                    if repo.branch not in remote_branchnames:
                        repo.info('Branch name not found in local remote. Attempting to fetch')
                        if dry:
                            repo.info('dry run, not fetching')
                        else:
                            repo._cmd('git fetch {}'.format(remote.name))
                            repo.info('Fetch was successful')
                else:
                    repo.debug('Requested remote does NOT exist')

            # Ensure the remote points to the right place
            if repo.url not in list(remote.urls):
                repo.debug('WARNING: The requested url={} disagrees with remote urls={}'.format(repo.url, list(remote.urls)))

                if dry:
                    repo.info('Dry run, not updating remote url')
                else:
                    repo.info('Updating remote url')
                    repo._cmd('git remote set-url {} {}'.format(repo.remote, repo.url))

            # Ensure we are on the right branch
            if repo.branch != repo.pygit.active_branch.name:
                repo.debug('NEED TO SET BRANCH TO {} for {}'.format(repo.branch, repo))
                if dry:
                    repo.info('Dry run, not setting branch')
                else:
                    try:
                        repo._cmd('git checkout {}'.format(repo.branch))
                    except ShellException:
                        repo.debug('Checkout failed. Branch name might be ambiguous. Trying again')
                        try:
                            repo._cmd('git fetch {}'.format(remote.name))
                            repo._cmd('git checkout -b {} {}/{}'.format(repo.branch, repo.remote, repo.branch))
                        except ShellException:
                            raise Exception('does the branch exist on the remote?')

            tracking_branch = repo.pygit.active_branch.tracking_branch()
            if tracking_branch is None or tracking_branch.remote_name != repo.remote:
                repo.debug('NEED TO SET UPSTREAM FOR FOR {}'.format(repo))

                try:
                    remote = repo.pygit.remotes[repo.remote]
                    if not remote.exists():
                        raise IndexError
                except IndexError:
                    repo.debug('WARNING: remote={} does not exist'.format(remote))
                else:
                    if remote.exists():
                        remote_branchnames = [ref.remote_head for ref in remote.refs]
                        if repo.branch not in remote_branchnames:
                            if dry:
                                repo.info('Branch name not found in local remote. Dry run, use ensure to attempt to fetch')
                            else:
                                repo.info('Branch name not found in local remote. Attempting to fetch')
                                repo._cmd('git fetch {}'.format(repo.remote))

                                remote_branchnames = [ref.remote_head for ref in remote.refs]
                                if repo.branch not in remote_branchnames:
                                    raise Exception('Branch name still does not exist')

                        if not dry:
                            repo._cmd('git branch --set-upstream-to={remote}/{branch} {branch}'.format(
                                remote=repo.remote, branch=repo.branch
                            ))
                        else:
                            repo.info('Would attempt to set upstream')

        # Print some status
        repo.debug(' * branch = {} -> {}'.format(
            repo.pygit.active_branch.name,
            repo.pygit.active_branch.tracking_branch(),
        ))

    def pull(repo):
        repo._assert_clean()
        # TODO: In past runs I've gotten the error:
        # Your configuration specifies to merge with the ref
        # 'refs/heads/dev/0.0.2' from the remote, but no such ref was fetched.
        # Doing an ensure seemed to fix it. We should do something to handle
        # this case ellegantly.
        repo._cmd('git pull')

    def status(repo):
        repo._cmd('git status')


def worker(repo, funcname, kwargs):
    repo.verbose = 0
    func = getattr(repo, funcname)
    func(**kwargs)
    return repo


class RepoRegistry(ub.NiceRepr):
    def __init__(registery, repos):
        registery.repos = repos

    def __nice__(registery):
        return ub.repr2(registery.repos, si=1, nl=1)

    def apply(registery, funcname, num_workers=0, **kwargs):
        print(ub.color_text('--- APPLY {} ---'.format(funcname), 'white'))
        print(' * num_workers = {!r}'.format(num_workers))

        if num_workers == 0:
            processed_repos = []
            for repo in registery.repos:
                print(ub.color_text('--- REPO = {} ---'.format(repo), 'blue'))
                try:
                    getattr(repo, funcname)(**kwargs)
                except DirtyRepoError:
                    print(ub.color_text('Ignoring dirty repo={}'.format(repo), 'red'))
                processed_repos.append(repo)
        else:
            from concurrent import futures
            # with futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            with futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
                tasks = []
                for i, repo in enumerate(registery.repos):
                    future = pool.submit(worker, repo, funcname, kwargs)
                    future.repo = repo
                    tasks.append(future)

                processed_repos = []
                for future in futures.as_completed(tasks):
                    repo = future.repo
                    print(ub.color_text('--- REPO = {} ---'.format(repo), 'blue'))
                    try:
                        repo = future.result()
                    except DirtyRepoError:
                        print(ub.color_text('Ignoring dirty repo={}'.format(repo), 'red'))
                    else:
                        print(repo._getlogs())
                    processed_repos.append(repo)

        print(ub.color_text('--- FINISHED APPLY {} ---'.format(funcname), 'white'))

        SHOW_CMDLOG = 1

        if SHOW_CMDLOG:

            print('LOGGED COMMANDS')
            import os
            ORIG_CWD = MY_CWD = os.getcwd()
            for repo in processed_repos:
                print('# --- For repo = {!r} --- '.format(repo))
                for t in repo._logged_cmds:
                    cmd, cwd = t
                    if cwd is None:
                        cwd = os.get_cwd()
                    if cwd != MY_CWD:
                        print('cd ' + ub.shrinkuser(cwd))
                        MY_CWD = cwd
                    print(cmd)
            print('cd ' + ub.shrinkuser(ORIG_CWD))


def determine_code_dpath():
    """
    Returns a good place to put the code for the internal dependencies.

    Returns:
        PathLike: the directory where you want to store your code

    In order, the methods used for determing this are:
        * the `--codedpath` command line flag (may be undocumented in the CLI)
        * the `--codedir` command line flag (may be undocumented in the CLI)
        * the CODE_DPATH environment variable
        * the CODE_DIR environment variable
        * the directory above this script (e.g. if this is in ~/code/repo/super_setup.py then code dir resolves to ~/code)
        * the user's ~/code directory.
    """
    import os
    candidates = [
        ub.argval('--codedir', default=''),
        ub.argval('--codedpath', default=''),
        os.environ.get('CODE_DPATH', ''),
        os.environ.get('CODE_DIR', ''),
    ]
    valid = [c for c in candidates if c != '']
    if len(valid) > 0:
        code_dpath = valid[0]
    else:
        try:
            # This file should be in the top level of a repo, the directory from
            # this file should be the code directory.
            this_fpath = abspath(__file__)
            code_dpath = abspath(dirname(dirname(this_fpath)))
        except NameError:
            code_dpath = ub.expandpath('~/code')

    if not exists(code_dpath):
        code_dpath = ub.expandpath(code_dpath)

    # if CODE_DIR and not exists(CODE_DIR):
    #     import warnings
    #     warnings.warn('environment variable CODE_DIR={!r} was defined, but does not exist'.format(CODE_DIR))

    if not exists(code_dpath):
        raise Exception(ub.codeblock(
            '''
            Please specify a correct code_dir using the CLI or ENV.
            code_dpath={!r} does not exist.
            '''.format(code_dpath)))
    return code_dpath


DEVEL_REPOS = [
    # The util libs
    {
        'name': 'kwarray', 'branch': 'dev/0.5.10', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/kwarray.git'},
    },
    {
        'name': 'kwimage', 'branch': 'dev/0.6.7', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/kwimage.git'},
    },
    {
        'name': 'kwannot', 'branch': 'dev/0.1.0', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/kwannot.git'},
    },
    {
        'name': 'kwcoco', 'branch': 'dev/0.1.7', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/kwcoco.git'},
    },
    {
        'name': 'kwplot', 'branch': 'dev/0.4.8', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/kwplot.git'},
    },

    # Pytorch deployer / exporter
    {
        'name': 'liberator', 'branch': 'dev/0.0.2', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:python/liberator.git'},
    },
    {
        'name': 'torch_liberator', 'branch': 'dev/0.0.5', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/torch_liberator.git'},
    },

    # For example data and CLI
    {
        'name': 'scriptconfig', 'branch': 'dev/0.5.8', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:utils/scriptconfig.git'},
    },
    {
        'name': 'ndsampler', 'branch': 'dev/0.5.12', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/ndsampler.git'},
    },

    # netharn - training harness
    {
        'name': 'netharn', 'branch': 'dev/0.5.10', 'remote': 'public',
        'remotes': {'public': 'git@gitlab.kitware.com:computer-vision/netharn.git'},
    },
]


def make_registry(devel_repos):
    code_dpath = determine_code_dpath()
    CommonRepo = functools.partial(Repo, code_dpath=code_dpath)
    repos = [CommonRepo(**kw) for kw in devel_repos]
    registery = RepoRegistry(repos)
    return registery


def main():
    devel_repos = DEVEL_REPOS
    registery = make_registry(devel_repos)

    only = ub.argval('--only', default=None)
    if only is not None:
        only = only.split(',')
        registery.repos = [repo for repo in registery.repos if repo.name in only]

    num_workers = int(ub.argval('--workers', default=8))
    if ub.argflag('--serial'):
        num_workers = 0

    protocol = ub.argval('--protocol', None)
    if ub.argflag('--https'):
        protocol = 'https'
    if ub.argflag('--http'):
        protocol = 'http'
    if ub.argflag('--ssh'):
        protocol = 'ssh'

    HACK_PROTOCOL = True
    if HACK_PROTOCOL:
        if protocol is None:
            # Try to determine if you are using ssh or https and default to that
            main_repo = None
            for repo in registery.repos:
                if repo.name == 'netharn':
                    main_repo = repo
                    break
            assert main_repo is not None
            for remote in repo.pygit.remotes:
                for url in list(remote.urls):
                    gurl1 = GitURL(url)
                    gurl2 = GitURL(repo.url)
                    if gurl2.parts()['path'] == gurl1.parts()['path']:
                        if gurl1.parts()['syntax'] == 'ssh':
                            protocol = 'ssh'
                        else:
                            protocol = 'https'
                        break
                if protocol is not None:
                    print('Found default protocol = {}'.format(protocol))
                    break

    if protocol is not None:
        for repo in registery.repos:
            repo.set_protocol(protocol)

    default_context_settings = {
        'help_option_names': ['-h', '--help'],
        'allow_extra_args': True,
        'ignore_unknown_options': True}

    @click.group(context_settings=default_context_settings)
    def cli_group():
        pass

    @cli_group.add_command
    @click.command('pull', context_settings=default_context_settings)
    def pull():
        registery.apply('pull', num_workers=num_workers)

    @cli_group.add_command
    @click.command('ensure', context_settings=default_context_settings)
    def ensure():
        """
        Ensure is the live run of "check".
        """
        registery.apply('ensure', num_workers=num_workers)

    @cli_group.add_command
    @click.command('ensure_clone', context_settings=default_context_settings)
    def ensure_clone():
        registery.apply('ensure_clone', num_workers=num_workers)

    @cli_group.add_command
    @click.command('check', context_settings=default_context_settings)
    def check():
        """
        Check is just a dry run of "ensure".
        """
        registery.apply('check', num_workers=num_workers)

    @cli_group.add_command
    @click.command('status', context_settings=default_context_settings)
    def status():
        registery.apply('status', num_workers=num_workers)

    @cli_group.add_command
    @click.command('develop', context_settings=default_context_settings)
    def develop():
        registery.apply('develop', num_workers=0)

    @cli_group.add_command
    @click.command('doctest', context_settings=default_context_settings)
    def doctest():
        registery.apply('doctest')

    @cli_group.add_command
    @click.command('versions', context_settings=default_context_settings)
    def versions():
        registery.apply('versions')

    cli_group()


_DOCKER_DEBUGGING = """
DOCKER_IMAGE=circleci/python
docker run -v $PWD:/io --rm -it $DOCKER_IMAGE bash

mkdir -p $HOME/code
cd $HOME/code
git clone -b dev/0.5.5 https://gitlab.kitware.com/computer-vision/netharn.git
cd $HOME/code/netharn

pip install -r requirements/super_setup.txt
python super_setup.py ensure --serial

"""


if __name__ == '__main__':
    """
    For autocomplete you must run in bash

    eval "$(_SUPER_SETUP_COMPLETE=source super_setup.py)"
    """
    # import click_completion
    # click_completion.init()
    main()
