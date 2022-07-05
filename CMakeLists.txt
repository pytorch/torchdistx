# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21.0)

project(torchdistx VERSION 0.3.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE RelWithDebInfo)
endif()

include(cmake/Helpers.cmake)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

include(CMakeDependentOption)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif()

option(TORCHDIST_BUILD_PYTHON "Generates build target for the Python C extension." ON)
option(TORCHDIST_BUILD_FOR_NATIVE "Builds for the processor type of the compiling machine." OFF)
option(TORCHDIST_TREAT_WARNINGS_AS_ERRORS "Treats compilation warnings as errors." OFF)
option(TORCHDIST_PERFORM_LTO "Performs link-time optimization." OFF)
option(TORCHDIST_INSTALL_STANDALONE "Installs with rpath." OFF)
option(TORCHDIST_RUN_CLANG_TIDY "Runs clang-tidy as static analyzer." OFF)

cmake_dependent_option(TORCHDIST_DEVELOP_PYTHON
    #DESCRIPTION
        "Copies the Python C extension to the source tree for `setup.py develop` mode."
    #VALUE
        ON
    #DEPENDS_ON
        TORCHDIST_BUILD_PYTHON
    #FORCE
        OFF
)

set(TORCHDIST_SANITIZERS
    #DEFAULT
        ""
    #TYPE
        CACHE STRING
    #DESCRIPTION
        "The list of sanitizers with which to build. The supported values are 'asan', 'ubsan', and 'tsan'."
)

set_property(CACHE TORCHDIST_SANITIZERS PROPERTY
    STRINGS
        "" "asan" "ubsan" "tsan"
)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

if(TORCHDIST_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    if(Python3_VERSION VERSION_LESS 3.7)
        message(FATAL_ERROR "Only CPython 3.7 and later versions are supported.")
    endif()

    # We have to ensure that we use a version of pybind11 that is ABI compatible
    # with PyTorch's version.
    torchdist_add_third_party(pybind11)
else()
    find_package(Python3 OPTIONAL COMPONENTS Interpreter)
endif()

if(Python3_EXECUTABLE)
    execute_process(
        COMMAND
            ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE
            torch_cmake_prefix_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(Torch 1.11 REQUIRED PATHS ${torch_cmake_prefix_path})

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

add_subdirectory(src/cc/torchdistx)

if(TORCHDIST_BUILD_PYTHON)
    add_subdirectory(src/python/torchdistx/_C)
endif()

torchdist_install_package(torchdistx
    #CONFIG_FILE
        src/cc/torchdistx-config.cmake.in
)

if(PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
    #TODO: Add catch2 tests.
endif()
