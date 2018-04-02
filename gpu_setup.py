from os.path import exists
from os.path import join
from os.path import os

CUDACONFIG = None


def find_in_path(name, path):
    """
    Find a file in a search path
    adapted fom
    http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    """
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    global CUDACONFIG

    # there doesnt seem to be an accepted standard for the CUDA envvar yet
    cuda_environs = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_SDK_ROOT_DIR', 'CUDAHOME']
    cuda_environs = [key for key in cuda_environs if key in os.environ]

    # first check for the env variable CUDA_HOME / CUDA_ROOT / etc.
    if cuda_environs:
        cuda_environ = cuda_environs[0]
        home = os.environ[cuda_environ]
        nvcc = join(home, 'bin', 'nvcc')
        if not exists(nvcc):
            raise EnvironmentError(
                'The nvcc binary={} does not exist in ${}'.format(
                    nvcc, cuda_environ))
    else:
        # otherwise, search the PATH for NVCC
        default_path = join(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc',
                            os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. '
                                   'Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': join(home, 'include'),
                  'lib64': join(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not '
                                   'be located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    if CUDACONFIG is None:
        locate_cuda()

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDACONFIG['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class TorchExtension(object):
    """
    customized hacked extension

    TODO: is there a better way to do this?
    """

    def __init__(self, name, sources, cuda_sources, extra_compile_args={},
                 with_cuda=True):
        self.name = name
        self.sources = sources
        self.cuda_sources = cuda_sources
        self.extra_compile_args = extra_compile_args
        self.with_cuda = with_cuda

    def build(self):
        from torch.utils.ffi import create_extension
        import ubelt as ub

        if CUDACONFIG is None:
            locate_cuda()

        sources = [p for p in self.sources if p.endswith('.c')]
        headers = [p for p in self.sources if p.endswith('.h')]
        cu_sources = [p for p in self.cuda_sources if p.endswith('.cu')]

        extra_objects = []
        defines = []
        if self.with_cuda:
            sources += [p for p in self.cuda_sources if p.endswith('.c')]
            headers += [p for p in self.cuda_sources if p.endswith('.h')]
            cu_objects = [p + '.o' for p in cu_sources]

            extra = ' '.join(self.extra_compile_args.get('nvcc', []))
            command_fmt = '{nvcc_exe} -c -o {cu_objects} {cu_sources} {extra}'
            command = command_fmt.format(
                nvcc_exe=CUDACONFIG['nvcc'],
                cu_objects=' '.join(cu_objects),
                cu_sources=' '.join(cu_sources),
                extra=extra,
            )
            info = ub.cmd(command, verbout=1, verbose=2)
            if info['ret'] != 0:
                raise Exception('Failed to build extension ' + self.name)

            for fpath in cu_objects:
                if not exists(fpath):
                    raise Exception('Object {} does not exist'.format(fpath))

            extra_objects += [os.path.abspath(p) for p in cu_objects]
            defines += [('WITH_CUDA', None)]

        ffi = create_extension(
            self.name,
            headers=headers,
            sources=sources,
            define_macros=defines,
            relative_to=__file__,
            with_cuda=self.with_cuda,
            extra_objects=extra_objects,
            # extra_compile_args=self.extra_compile_args
        )
        ffi.build()
