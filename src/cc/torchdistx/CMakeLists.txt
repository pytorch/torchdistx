# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

torchdist_add_target(torchdistx SHARED_LIBRARY)

add_library(torchdistx::torchdistx ALIAS torchdistx)

target_sources(torchdistx
    PRIVATE
        deferred_init.cc
        fake.cc
        stack_utils.cc
)

target_compile_features(torchdistx
    PUBLIC
        cxx_std_17
)

target_link_libraries(torchdistx
    PUBLIC
        torch
    PRIVATE
        Threads::Threads
)

torchdist_install(torchdistx
    PACKAGE
        torchdistx
    HEADERS
        deferred_init.h
        fake.h
        macros.h
)
