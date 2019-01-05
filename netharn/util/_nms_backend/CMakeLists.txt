
option(BUILD_CPU_NMS "Cython CPU-based NMS" True)
if (BUILD_CPU_NMS)
  set(cython_source "cpu_nms.pyx")
  set(module_name "cpu_nms")

  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_source}" C OUTPUT_VAR sources)
  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  target_include_directories(${module_name} PUBLIC ${NumPy_INCLUDE_DIRS})
  target_include_directories(${module_name} PUBLIC ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(${module_name} ${PYTHON_LIBRARIES})
  # Transform the C++ library into an importable python module
  python_extension_module(${module_name})
  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
endif()



option(BUILD_GPU_NMS "Cython GPU-based NMS" True)
if (BUILD_GPU_NMS)
  set(cython_source "gpu_nms.pyx")
  set(module_name "gpu_nms")

  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_source}" CXX OUTPUT_VAR sources)

  # Add any other non-cython dependencies to the sources
  list(APPEND sources "nms_kernel.cu")
  list(APPEND sources "gpu_nms.hpp")

  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  target_include_directories(${module_name} PUBLIC
    ${NumPy_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${module_name} ${CUDA_LIBRARIES} ${PYTHON_LIBRARIES})

  # Transform the C++ library into an importable python module
  python_extension_module(${module_name})

  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
endif()

