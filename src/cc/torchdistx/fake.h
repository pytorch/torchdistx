// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include <c10/core/DispatchKey.h>
#include <c10/core/Storage.h>

#include "macros.h"

namespace at {

class TensorBase;

}  // namespace at

namespace torchdistx {
namespace detail {

// Since as of this implementation PyTorch is out of dispatch keys we hijack
// the dispatch key of functorch. The implication of this workaround is that
// functorch and fake tensors cannot be used in the same process.
//
// TODO: Once the dispatch key limitation is resolved define our own key.
constexpr auto kFakeDispatchKey = at::DispatchKey::FuncTorchDynamicLayerBackMode;

}  // namespace detail

// Forces all newly-constructed tensors on the calling thread to be fake.
TDX_API void enableFakeMode(bool value);

// Indicates whether `tensor` is fake.
TDX_API bool isFake(const at::TensorBase& tensor) noexcept;

// Returns the meta storage of `fake`.
TDX_API const at::Storage& getFakeMetaStorage(const at::TensorBase& fake);

// Stores an opaque context object for `key` in `fake`.
TDX_API void setFakeContext(at::TensorBase& fake, at::DispatchKey key, std::shared_ptr<void> ctx);

// Determines whether `key` has a context object in `fake`.
TDX_API bool hasFakeContext(const at::TensorBase& fake, at::DispatchKey key);

// Retrieves the context object of `key` from `fake`.
TDX_API std::shared_ptr<void> getFakeContext(const at::TensorBase& fake, at::DispatchKey key);

// Retrieves the context object of `key` from `fake`.
template <typename ContextType>
TDX_API inline auto getFakeContext(const at::TensorBase& fake, at::DispatchKey key) {
  return std::static_pointer_cast<ContextType>(getFakeContext(fake, key));
}

// Retrieves the context object of `key` from `fake`.
TDX_API void* unsafeGetFakeContext(const at::TensorBase& fake, at::DispatchKey key);

// Retrieves the context object of `key` from `fake`.
template <typename ContextType>
TDX_API inline auto unsafeGetFakeContext(const at::TensorBase& fake, at::DispatchKey key) {
  return static_cast<ContextType*>(unsafeGetFakeContext(fake, key));
}

}  // namespace torchdistx
