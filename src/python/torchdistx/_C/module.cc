// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "module.h"

#include <exception>

#include <torch/csrc/Exceptions.h>

namespace py = pybind11;

namespace torchdistx::python {
namespace {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void registerLocalExceptionTranslator() {
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  py::register_local_exception_translator([](std::exception_ptr ex) {
    try {
      if (ex) {
        std::rethrow_exception(ex);
      }
    }
    CATCH_TH_ERRORS()  // NOLINT
  });
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  registerLocalExceptionTranslator();

  initDeferredInitFunctions(m);

  initFakeFunctions(m);
}

}  // namespace torchdistx::python
