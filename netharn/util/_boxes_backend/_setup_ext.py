

def gather_extensions():
    """
    Used by setup.py Returns the ext_modules that we will pass to setup
    """
    from os.path import join
    import numpy as np
    from distutils.extension import Extension
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    ext_modules = [
        Extension(
            # TODO: maybe generalize the root of this
            'kwil.structs._boxes_backend.cython_boxes',
            [join('kwil/structs/_boxes_backend/cython_boxes.pyx')],
            extra_compile_args={'gcc': ['-Wno-cpp', '-Wno-unused-function']},
            include_dirs=[numpy_include]
        )
    ]
    return ext_modules
