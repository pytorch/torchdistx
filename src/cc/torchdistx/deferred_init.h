// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include "macros.h"

namespace at {

class Tensor;

}  // namespace at

namespace torchdistx {

// Forces all newly-constructed tensors on the calling thread to be fake while
// also recording all operations performed on them in memory. Such tensors can
// later be materialized by calling `materializeTensor()`.
TDX_API void enableDeferredInit(bool value);

// Materializes `tensor`.
TDX_API at::Tensor materializeTensor(const at::Tensor& tensor);

// Temporarily disables deferred-init.
class TDX_API NoDeferredInit {
  c10::impl::ExcludeDispatchKeyGuard guard_{at::DispatchKey::DeferredInit};
};

}  // namespace torchdistx
