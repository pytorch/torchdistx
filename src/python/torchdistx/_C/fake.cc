// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "module.h"

#include <ATen/Tensor.h>
#include <torch/csrc/utils/pybind.h>
#include <torchdistx/fake.h>

namespace torchdistx::python {

void initFakeFunctions(pybind11::module& m) {
  m.def("enable_fake_mode", enableFakeMode);

  m.def("is_fake", [](const at::Tensor& tensor) {
    return isFake(tensor);  // cast to `TensorBase`.
  });
}

}  // namespace torchdistx::python
