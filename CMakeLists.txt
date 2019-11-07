###
# CMakeLists for Netharn C/C++/Cuda Python extention modules
#
# Notes:
# https://github.com/Erotemic/netharn/issues/7
# https://github.com/amueller/word_cloud/pull/42ci

cmake_minimum_required(VERSION 3.11.0)
project(netharn LANGUAGES C CXX CUDA)


# Setup basic python stuff and ensure we have skbuild
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/CMake")
###
# Finds the python binaries, libs, include, and site-packages paths
# Ensures that skbuild is installed and its utilities are findable
#
# Calls find_packages to on python interpreter/libraries which defines:
#
#    PYTHON_EXECUTABLE
#    PYTHON_INCLUDE_DIR
#    PYTHON_LIBRARY
#    PYTHON_LIBRARY_DEBUG
#
# Exported variables used by python utility functions are:
#
#    skbuild_location
#      Location of the skbuild library (assumes you have run `pip install scikit-build`)
#
#    skbuild_cmake_dir
#      Location of the skbuild cmake utilities


###
# Private helper function to execute `python -c "<cmd>"`
#
# Runs a python command and populates an outvar with the result of stdout.
# Be careful of indentation if `cmd` is multiline.
#
function(pycmd outvar cmd)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "${cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  if(NOT ${_exitcode} EQUAL 0)
    message(ERROR "Failed when running python code: \"\"\"
${cmd}\"\"\"")
    message(FATAL_ERROR "Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
endfunction()


###
# Find current python major version user option
#

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})


###
# Find scikit-build and include its cmake resource scripts
#
if (NOT SKBUILD)
  pycmd(skbuild_location "import os, skbuild; print(os.path.dirname(skbuild.__file__))")
  set(skbuild_cmake_dir "${skbuild_location}/resources/cmake")
  # If skbuild is not the driver, then we need to include its utilities in our CMAKE_MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH ${skbuild_cmake_dir})
endif()



###
# Status string for debugging
#
set(PYTHON_SETUP_STATUS "
  * PYTHON_EXECUTABLE = \"${PYTHON_EXECUTABLE}\"

  * PYTHON_INCLUDE_DIR = \"${PYTHON_INCLUDE_DIR}\"
  * PYTHON_LIBRARY = \"${PYTHON_LIBRARY}\"
  * PYTHON_LIBRARY_DEBUG = \"${PYTHON_LIBRARY_DEBUG}\"

  * skbuild_location = \"${skbuild_location}\"
  * skbuild_cmake_dir = \"${skbuild_cmake_dir}\"
")


option(BUILD_CUDA_LIB "Build with CUDA" "True")
if (BUILD_CUDA_LIB)
    find_package(CUDA)
    if (CUDA_FOUND)
        message(STATUS "CUDA found (include: ${CUDA_INCLUDE_DIRS}, lib: ${CUDA_LIBRARIES})")
        include_directories(${CUDA_INCLUDE_DIRS})
    else(CUDA_FOUND)
        message(STATUS "CUDA not found, CUDA library will not be built")
        set(BUILD_CUDA_LIB OFF)
    endif(CUDA_FOUND)
endif(BUILD_CUDA_LIB)


option(NETHARN_VERBOSE "Print extra info" False)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)


add_subdirectory("netharn/util/_boxes_backend")
add_subdirectory("netharn/util/_nms_backend")


set(NETHARN_CONFIG_STATUS "
PYTHON_CONFIG_STATUS
====================

Include Dirs:
  * CUDA_INCLUDE_DIRS = \"${CUDA_INCLUDE_DIRS}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * PYTHON_INCLUDE_DIRS = \"${PYTHON_INCLUDE_DIRS}\"

status(Cython):
  * CYTHON_FOUND = \"${CYTHON_FOUND}\"
  * CYTHON_EXECUTABLE = \"${CYTHON_EXECUTABLE}\"
  * CYTHON_VERSION = \"${CYTHON_VERSION}\"

status(NumPy):
  * NumPy_FOUND = \"${NumPy_FOUND}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * NumPy_VERSION = \"${NumPy_VERSION}\"
  * NumPy_CONV_TEMPLATE_EXECUTABLE = \"${NumPy_CONV_TEMPLATE_EXECUTABLE}\"
  * NumPy_FROM_TEMPLATE_EXECUTABLE = \"${NumPy_FROM_TEMPLATE_EXECUTABLE}\"

status(PythonExtensions):
  * PYTHON_PREFIX = \"${PYTHON_PREFIX}\"
  * PYTHON_SITE_PACKAGES_DIR = \"${PYTHON_SITE_PACKAGES_DIR}\"
  * PYTHON_RELATIVE_SITE_PACKAGES_DIR = \"${PYTHON_RELATIVE_SITE_PACKAGES_DIR}\"
  * PYTHON_SEPARATOR = \"${PYTHON_SEPARATOR}\"
  * PYTHON_PATH_SEPARATOR = \"${PYTHON_PATH_SEPARATOR}\"
  * PYTHON_EXTENSION_MODULE_SUFFIX = \"${PYTHON_EXTENSION_MODULE_SUFFIX}\"

status(python-setup)
${PYTHON_SETUP_STATUS}
")

if (NETHARN_VERBOSE)
  message(STATUS ${NETHARN_CONFIG_STATUS})
endif ()
