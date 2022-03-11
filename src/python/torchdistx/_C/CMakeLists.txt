# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

torchdist_add_target(torchdistx-py PYTHON_MODULE
    OUTPUT_NAME
        _C
)

target_sources(torchdistx-py
    PRIVATE
        deferred_init.cc
        fake.cc
        module.cc
)

target_compile_features(torchdistx-py
    PRIVATE
        cxx_std_17
)

cmake_path(GET
    #VAR
        TORCH_LIBRARY
    PARENT_PATH
        torch_library_dir
)

# libtorch_python is not exported as part of Torch CMake package, so we have to
# manually find it.
find_library(TORCH_PYTHON_LIBRARY
    #NAME
        torch_python
    PATHS
        ${torch_library_dir}
)

mark_as_advanced(TORCH_PYTHON_LIBRARY)

target_link_libraries(torchdistx-py
    PRIVATE
        pybind11::module torch torchdistx ${TORCH_PYTHON_LIBRARY}
)

torchdist_install_python_module(torchdistx-py)
