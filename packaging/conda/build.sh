#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -o errexit

cmake -GNinja\
      -DCMAKE_BUILD_TYPE=RelWithDebInfo\
      -DCMAKE_INSTALL_PREFIX="$PREFIX"\
      -DCMAKE_INSTALL_LIBDIR=lib\
      -DCMAKE_FIND_FRAMEWORK=NEVER\
      -DTORCHDIST_TREAT_WARNINGS_AS_ERRORS=ON\
      -DTORCHDIST_PERFORM_LTO=ON\
      -DTORCHDIST_DEVELOP_PYTHON=OFF\
      -S "$SRC_DIR"\
      -B "$SRC_DIR/build"

cmake --build "$SRC_DIR/build"

# Extract the debug symbols; they will be part of the debug package.
find "$SRC_DIR/build" -type f -name "libtorchdistx*"\
    -exec "$SRC_DIR/scripts/strip-debug-symbols" --extract "{}" ";"
