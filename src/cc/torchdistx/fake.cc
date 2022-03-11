// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fake.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/python_stub.h>
#include <torch/library.h>

#include "stack_utils.h"

namespace torchdistx {

using at::Argument;
using at::Device;
using at::DispatchKey;
using at::DispatchKeySet;
using at::getAutocastRelatedKeySetFromBackend;
using at::getAutogradRelatedKeySetFromBackend;
using at::intrusive_ptr;
using at::IValue;
using at::nullopt;
using at::OperatorHandle;
using at::optional;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::TensorBase;
using at::TensorImpl;
using at::TorchDispatchTypeObject;
using at::typeMetaToScalarType;
using at::VariableVersion;

using c10::impl::ExcludeDispatchKeyGuard;
using c10::impl::PyInterpreter;
using c10::impl::PyInterpreterStatus;
using c10::impl::tls_set_dispatch_key_included;

using torch::jit::Stack;

};  // namespace torchdistx

namespace torchdistx::detail {
namespace {

// See the note [Meta to Fake Tensor] to understand why we have a non-functional
// Python interpreter here.
class FakePyInterpreter {
 private:
  [[noreturn]] static std::string getName(const PyInterpreter*) {
    failCall("name");
  }

  [[noreturn]] static void decref(const PyInterpreter*, PyObject*, bool) {
    failCall("decref");
  }

  [[noreturn]] static intrusive_ptr<TensorImpl> detach(const PyInterpreter*, const TensorImpl*) {
    failCall("detach");
  }

  [[noreturn]] static void dispatch(const PyInterpreter*, const OperatorHandle&, Stack*,
                                    const std::shared_ptr<TorchDispatchTypeObject>&) {
    failCall("dispatch");
  }

  [[noreturn]] static void failCall(const char* function_name) {
    TORCH_INTERNAL_ASSERT(false,
        "`FakePyInterpreter::", function_name, "()` run unexpectedly.");
  }

  FakePyInterpreter() noexcept : impl_{getName, decref, detach, dispatch} {}

 public:
  FakePyInterpreter(FakePyInterpreter&) = delete;

  FakePyInterpreter& operator=(FakePyInterpreter&) = delete;

  FakePyInterpreter(FakePyInterpreter&&) = delete;

  FakePyInterpreter& operator=(FakePyInterpreter&&) = delete;

  ~FakePyInterpreter() {
    impl_.disarm();
  }

 public:
  static PyInterpreter* get() noexcept {
    static FakePyInterpreter singleton{};

    return &singleton.impl_;
  }

 private:
  PyInterpreter impl_;
};

// A fake tensor acts very much like an opaque tensor (i.e. `OpaqueTensorImpl`)
// to the dispatch keys above `Fake`. This means it has no storage allocated to
// it, but still resides on a real device. However, unlike an opaque tensor, it
// internally holds a meta tensor that is used for the actual dispatch.
class FakeTensorImpl : public TensorImpl {
  // Let `make_intrusive()` access our private constructor.
  friend class intrusive_ptr<FakeTensorImpl>;

 private:
  // Constructs an empty instance. It is private since `makeFromMeta()` is the
  // actual factory function for fake tensors.
  explicit FakeTensorImpl() noexcept : TensorImpl{DispatchKeySet{}, caffe2::TypeMeta{}, nullopt} {}

  static DispatchKeySet computeFakeKeySet(TensorImpl& meta_impl, Device fake_device);

  void shallowCopyFromMeta(const TensorImpl& meta_impl, Device fake_device,
                           DispatchKeySet fake_key_set);

 public:
  // Copies all metadata of `meta_impl` except its storage and device.
  void shallowCopyFromMeta(const TensorImpl& meta_impl);

  // Constructs a new fake tensor by copying the metadata of `meta_impl` and
  // using `fake_device` as the device.
  static intrusive_ptr<FakeTensorImpl> makeFromMeta(intrusive_ptr<TensorImpl> meta_impl,
                                                    Device fake_device);

  // Returns the fake tensor that is holding `meta_impl` or null if `meta_impl`
  // is not owned by a fake tensor.
  static intrusive_ptr<FakeTensorImpl> tryGetFromMeta(TensorImpl& meta_impl) noexcept;

 public:
  void shallow_copy_from(const intrusive_ptr<TensorImpl>& impl) override;

  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_metadata_change) const override;

  intrusive_ptr<TensorImpl> shallow_copy_and_detach(VariableVersion&& version_counter,
                                                    bool allow_metadata_change) const override;

  void release_resources() override;

 protected:
  const char* tensorimpl_type_name() const override {
    return "FakeTensorImpl";
  }

 public:
  const intrusive_ptr<TensorImpl>& meta_impl() const noexcept {
    return meta_impl_;
  }

  // Each dispatch handler can have its own contextual data associated with a
  // fake tensor. For example the `DeferredInit` handler stores the operation
  // graph node that output the fake tensor in this map.
  std::unordered_map<DispatchKey, std::shared_ptr<void>> dispatch_context{};

 private:
  // Assigns `meta_impl` as the meta tensor of this instance.
  void setMeta(intrusive_ptr<TensorImpl>&& meta_impl);

  // The meta tensor that this instance is holding. It is used for diverting
  // operators to the meta backend.
  intrusive_ptr<TensorImpl> meta_impl_;
};

DispatchKeySet FakeTensorImpl::computeFakeKeySet(TensorImpl& meta_impl, Device fake_device) {
  ScalarType data_type = typeMetaToScalarType(meta_impl.dtype());

  // We use the data type and layout of `meta_impl`, but use `fake_device`
  // instead of the meta device to compute the backend dispatch key.
  DispatchKey backend_key = computeDispatchKey(data_type, meta_impl.layout(), fake_device);

  // We also mix the `Fake` dispatch key to ensure that the Fake handler gets
  // called instead of the actual backend handler.
  DispatchKeySet key_set{backend_key, kFakeDispatchKey};

  if (meta_impl.is_inference()) {
    return key_set;
  }

  key_set = key_set | getAutocastRelatedKeySetFromBackend(backend_key);
  key_set = key_set | getAutogradRelatedKeySetFromBackend(backend_key);

  return key_set;
}

void FakeTensorImpl::shallowCopyFromMeta(const TensorImpl& meta_impl, Device fake_device,
                                         DispatchKeySet fake_key_set) {
  copy_tensor_metadata(&meta_impl, this, version_counter_, allow_tensor_metadata_change_);

  // Do not allow `copy_tensor_metadata()` to set the storage and the device of
  // our instance. Ensure that we continue to act like an opaque tensor.
  storage_ = {};

  storage_access_should_throw_ = true;

  device_opt_ = fake_device;

  key_set_ = fake_key_set;
}

void FakeTensorImpl::shallowCopyFromMeta(const TensorImpl& meta_impl) {
  TORCH_INTERNAL_ASSERT(meta_impl.is_meta(),
      "The source tensor was expected to be a meta tensor.");

  shallowCopyFromMeta(meta_impl, *device_opt_, key_set_);

  refresh_numel();
  refresh_contiguous();
}

intrusive_ptr<FakeTensorImpl> FakeTensorImpl::makeFromMeta(intrusive_ptr<TensorImpl> meta_impl,
                                                           Device fake_device) {
  TORCH_INTERNAL_ASSERT(meta_impl->is_meta(),
      "The source tensor was expected to be a meta tensor.");

  DispatchKeySet fake_key_set = computeFakeKeySet(*meta_impl, fake_device);

  auto fake_impl = at::make_intrusive<FakeTensorImpl>();

  fake_impl->shallowCopyFromMeta(*meta_impl, fake_device, fake_key_set);

  fake_impl->refresh_numel();
  fake_impl->refresh_contiguous();

  fake_impl->setMeta(std::move(meta_impl));

  return fake_impl;
}

intrusive_ptr<FakeTensorImpl> FakeTensorImpl::tryGetFromMeta(TensorImpl& meta_impl) noexcept {
  // Note [Meta to Fake Tensor]
  // To access the fake tensor holding `meta_impl` we take advantage of the
  // `pyobj_` field of `TensorImpl`. Since our meta tensor is never exposed
  // to Python it is safe to use its `pyobj_` field to store a pointer back
  // to its fake tensor. This way we can access the fake tensor in constant
  // time without using any other data structure.
  //
  // Our `FakePyInterpreter` has no purpose other then sanity check. We use
  // it solely to retrieve the fake tensor and any call to it will cause an
  // assertion failure.
  optional<PyObject*> opt_fake_py_obj = meta_impl.check_pyobj(FakePyInterpreter::get());
  if (opt_fake_py_obj != nullopt) {
    // We lie about `pyobj_` being a `PyObject`. It actually points to the fake
    // tensor owning `meta_impl`.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* fake_impl_ptr = reinterpret_cast<FakeTensorImpl*>(*opt_fake_py_obj);

    return intrusive_ptr<FakeTensorImpl>::reclaim_copy(fake_impl_ptr);
  } else {
    return {};
  }
}

void FakeTensorImpl::shallow_copy_from(const intrusive_ptr<TensorImpl>& impl) {
  TORCH_CHECK(impl->key_set().has(kFakeDispatchKey),
      "The source tensor was expected to be a fake tensor.");

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  const auto* src_impl = static_cast<const FakeTensorImpl*>(impl.get());

  copy_tensor_metadata(src_impl, this, version_counter_, allow_tensor_metadata_change_);

  refresh_numel();
  refresh_contiguous();

  meta_impl_->shallow_copy_from(src_impl->meta_impl_);
}

intrusive_ptr<TensorImpl> FakeTensorImpl::shallow_copy_and_detach(
    const VariableVersion& version_counter, bool allow_metadata_change) const {
  auto impl = at::make_intrusive<FakeTensorImpl>();

  copy_tensor_metadata(this, impl.get(), version_counter, allow_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();

  intrusive_ptr<TensorImpl> meta_impl = meta_impl_->shallow_copy_and_detach(0, false);

  impl->setMeta(std::move(meta_impl));

  return impl;
}

intrusive_ptr<TensorImpl> FakeTensorImpl::shallow_copy_and_detach(
    VariableVersion&& version_counter, bool allow_metadata_change) const {
  auto impl = at::make_intrusive<FakeTensorImpl>();

  copy_tensor_metadata(this, impl.get(), std::move(version_counter), allow_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();

  intrusive_ptr<TensorImpl> meta_impl = meta_impl_->shallow_copy_and_detach(0, false);

  impl->setMeta(std::move(meta_impl));

  return impl;
}

void FakeTensorImpl::release_resources() {
  TensorImpl::release_resources();

  meta_impl_ = {};

  dispatch_context.clear();
}

void FakeTensorImpl::setMeta(intrusive_ptr<TensorImpl>&& impl) {
  constexpr auto s = PyInterpreterStatus::DEFINITELY_UNINITIALIZED;

  // See the note [Meta to Fake Tensor] to understand why we lie about our
  // instance being a Python object.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  impl->init_pyobj(FakePyInterpreter::get(), reinterpret_cast<PyObject*>(this), s);

  meta_impl_ = std::move(impl);
}

// Returns the meta tensor held by `fake`.
inline Tensor getMetaTensor(const Tensor& fake) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  const auto* fake_impl = static_cast<const FakeTensorImpl*>(fake.unsafeGetTensorImpl());

  return Tensor::wrap_tensor_impl(fake_impl->meta_impl());
}

// The catch-all handler for the `Fake` dispatch key.
class FakeHandler {
 public:
  explicit FakeHandler(const OperatorHandle& op, DispatchKeySet key_set, Stack* s) noexcept
      : handle_{&op}, key_set_{key_set}, stack_{s} {}

  void run();

 private:
  void assessOp();

  optional<Device> inferDeviceOfTensorArguments();

  IValue* getDeviceArgumentPosition() const noexcept;

  bool hasBackendSelectKernel() const noexcept;

  bool hasTensorOptionsInArguments() const noexcept;

  Device determineOutputDevice() const;

  void convertFakeArgumentsToMetaTensors();

  bool shouldFakeOp() const;

  void convertDeviceArgumentToMeta() noexcept;

  void redispatchToMeta();

  bool hasKernelForDispatchKey(DispatchKey) const noexcept;

  void convertMetaOutputsToFakeTensors();

  void convertToFakeTensor(Tensor& tensor);

  void redispatchToBackend();

 private:
  static const DispatchKeySet kAfterFakeKeySet_;

  const OperatorHandle* handle_;
  DispatchKeySet key_set_;
  Stack* stack_;
  optional<Device> opt_inferred_device_{};
  IValue* device_arg_pos_{};
  Device output_device_ = at::kCPU;
  bool has_fake_ = false;
  bool has_tensor_arg_ = false;
};

// NOLINTNEXTLINE(cert-err58-cpp)
const DispatchKeySet FakeHandler::kAfterFakeKeySet_{DispatchKeySet::FULL_AFTER, kFakeDispatchKey};

inline bool isCPUScalar(const Tensor& tensor) noexcept {
  return tensor.dim() == 0 && tensor.is_cpu();
}

void FakeHandler::run() {
  ExcludeDispatchKeyGuard guard{kFakeDispatchKey};

  assessOp();

  convertFakeArgumentsToMetaTensors();

  // We divert the operator to the meta backend only if it is a factory or if it
  // has a fake tensor argument; otherwise we call the original backend.
  if (shouldFakeOp()) {
    convertDeviceArgumentToMeta();

    redispatchToMeta();

    convertMetaOutputsToFakeTensors();
  } else {
    redispatchToBackend();
  }
}

void FakeHandler::assessOp() {
  opt_inferred_device_ = inferDeviceOfTensorArguments();

  device_arg_pos_ = getDeviceArgumentPosition();

  output_device_ = determineOutputDevice();
}

optional<Device> FakeHandler::inferDeviceOfTensorArguments() {
  optional<Device> opt_device{};

  auto fn = [&opt_device](const Tensor& tensor) {
    if (isCPUScalar(tensor)) {
      return false;
    }

    if (opt_device != nullopt) {
      TORCH_CHECK(*opt_device == tensor.device(),
          "Expected all tensors to be on the same device, but found at least two devices, ",
          *opt_device, " and ", tensor.device(), "!");
    } else {
      opt_device = tensor.device();
    }

    return false;
  };

  processTensors(*stack_, handle_->schema().arguments().size(), fn);

  return opt_device;
}

IValue* FakeHandler::getDeviceArgumentPosition() const noexcept {
  // Having a parameter named `device` by itself is not sufficient to conclude
  // that it specifies the desired output device. We also use a heuristic that
  // checks the operator and its arguments.
  if (hasBackendSelectKernel() || hasTensorOptionsInArguments()) {
    const std::vector<Argument>& args = handle_->schema().arguments();
    for (auto pos = args.begin(); pos < args.end(); ++pos) {
      if (pos->name() == "device") {
        return &torch::jit::peek(stack_, static_cast<std::size_t>(pos - args.begin()), args.size());
      }
    }
  }
  return nullptr;
}

inline bool FakeHandler::hasBackendSelectKernel() const noexcept {
  return handle_->hasKernelForDispatchKey(DispatchKey::BackendSelect);
}

bool FakeHandler::hasTensorOptionsInArguments() const noexcept {
  std::array<at::string_view, 4> tensor_opts = {{"dtype", "layout", "device", "pin_memory"}};

  const std::vector<Argument>& args = handle_->schema().arguments();
  if (args.size() < tensor_opts.size()) {
    return false;
  }

  // Checks if the arguments starting at `arg_pos` represent a `TensorOptions`.
  auto are_tensor_opts = [&tensor_opts](auto arg_pos) noexcept {
    for (const auto& tensor_opt : tensor_opts) {
      if (tensor_opt != arg_pos->name()) {
        return false;
      }
      ++arg_pos;
    }
    return true;
  };

  for (auto pos = args.begin(); pos <= args.end() - tensor_opts.size(); ++pos) {
    if (are_tensor_opts(pos)) {
      return true;
    }
  }
  return false;
}

// TODO: Note that this implementation is a simple heuristic and can fail to
// determine the real output device. In the future we should use a mechanism
// that is more robust (e.g. operator tagging).
Device FakeHandler::determineOutputDevice() const {
  // Use the explicitly specified `device` argument.
  if (device_arg_pos_ != nullptr && device_arg_pos_->isDevice()) {
    return device_arg_pos_->toDevice();

    // Otherwise; use the device of the first tensor argument.
  } else if (opt_inferred_device_ != nullopt) {
    return *opt_inferred_device_;

    // Otherwise; fallback to CPU.
  } else {
    return at::kCPU;
  }
}

void FakeHandler::convertFakeArgumentsToMetaTensors() {
  auto fn = [this](Tensor& tensor) {
    if (isFake(tensor)) {
      tensor = getMetaTensor(tensor);

      has_fake_ = true;
    }

    has_tensor_arg_ = true;

    return false;
  };

  convertTensors(*stack_, handle_->schema().arguments().size(), fn);
}

inline bool FakeHandler::shouldFakeOp() const {
  return has_fake_ || device_arg_pos_ != nullptr || !has_tensor_arg_;
}

void FakeHandler::convertDeviceArgumentToMeta() noexcept {
  if (device_arg_pos_ == nullptr) {
    return;
  }

  IValue& device_arg = *device_arg_pos_;

  device_arg = Device{at::kMeta};
}

void FakeHandler::redispatchToMeta() {
  auto backend_key = (key_set_ & kAfterFakeKeySet_).highestPriorityBackendTypeId();

  if (backend_key != DispatchKey::Undefined) {
    TORCH_CHECK_NOT_IMPLEMENTED(hasKernelForDispatchKey(backend_key),
        "The dispatch key `", backend_key, "` has no kernel for `", handle_->schema().name(), "`.");
  }

  TORCH_CHECK_NOT_IMPLEMENTED(hasKernelForDispatchKey(DispatchKey::Meta),
      "`", handle_->schema().name(), "` cannot be run with fake tensor(s) because the meta backend "
      "has no kernel for it. Please file an issue if you want it to be supported.");

  handle_->redispatchBoxed(DispatchKeySet(DispatchKey::Meta), stack_);
}

bool FakeHandler::hasKernelForDispatchKey(DispatchKey key) const noexcept {
  return handle_->hasKernelForDispatchKey(key) ||
         handle_->hasKernelForDispatchKey(DispatchKey::CompositeExplicitAutograd) ||
         handle_->hasKernelForDispatchKey(DispatchKey::CompositeImplicitAutograd);
}

void FakeHandler::convertMetaOutputsToFakeTensors() {
  auto fn = [this](Tensor& tensor) {
    convertToFakeTensor(tensor);
  };

  convertTensors(*stack_, handle_->schema().returns().size(), fn);
}

void FakeHandler::convertToFakeTensor(Tensor& tensor) {
  const intrusive_ptr<TensorImpl>& meta_impl = tensor.getIntrusivePtr();

  intrusive_ptr<FakeTensorImpl> fake_impl = FakeTensorImpl::tryGetFromMeta(*meta_impl);
  // If `fake_impl` is not null, it means we had an in-place operator that
  // returned one of its tensor arguments.
  if (fake_impl) {
    // Ensure that we reflect any changes to the meta tensor's metadata such as
    // shape or layout changes to the fake tensor.
    fake_impl->shallowCopyFromMeta(*meta_impl);
  } else {
    fake_impl = FakeTensorImpl::makeFromMeta(meta_impl, output_device_);
  }

  tensor = Tensor::wrap_tensor_impl(std::move(fake_impl));
}

void FakeHandler::redispatchToBackend() {
  handle_->redispatchBoxed(key_set_ & kAfterFakeKeySet_, stack_);
}

void runFakeHandler(const OperatorHandle& op, DispatchKeySet key_set, Stack* s) {
  FakeHandler{op, key_set, s}.run();
}

}  // namespace
}  // namespace torchdistx::detail

// NOLINTNEXTLINE(cert-err58-cpp)
TORCH_LIBRARY_IMPL(_, /*Fake*/ FuncTorchDynamicLayerBackMode, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&torchdistx::detail::runFakeHandler>());
}

namespace torchdistx {
namespace detail {
namespace {

FakeTensorImpl* getFakeTensorImpl(const TensorBase& tensor) {
  TORCH_CHECK(isFake(tensor),
      "`fake` was expected to be a fake tensor.");

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
  return static_cast<detail::FakeTensorImpl*>(tensor.unsafeGetTensorImpl());
}

}  // namespace
}  // namespace detail

namespace {

thread_local std::size_t tls_fake_mode_level = 0;

}  // namespace

void enableFakeMode(bool value) {
  if (value) {
    if (tls_fake_mode_level++ == 0) {
      tls_set_dispatch_key_included(detail::kFakeDispatchKey, value);
    }
  } else if (tls_fake_mode_level > 0) {
    if (tls_fake_mode_level-- == 1) {
      tls_set_dispatch_key_included(detail::kFakeDispatchKey, value);
    }
  }
}

bool isFake(const TensorBase& tensor) noexcept {
  return tensor.key_set().has(detail::kFakeDispatchKey);
}

const Storage& getFakeMetaStorage(const TensorBase& fake) {
  return detail::getFakeTensorImpl(fake)->meta_impl()->storage();
}

void setFakeContext(TensorBase& fake, DispatchKey key, std::shared_ptr<void> ctx) {
  if (ctx) {
    detail::getFakeTensorImpl(fake)->dispatch_context.insert_or_assign(key, std::move(ctx));
  } else {
    detail::getFakeTensorImpl(fake)->dispatch_context.erase(key);
  }
}

bool hasFakeContext(const TensorBase& fake, DispatchKey key) {
  auto& ctx = detail::getFakeTensorImpl(fake)->dispatch_context;

  return ctx.find(key) != ctx.end();
}

std::shared_ptr<void> getFakeContext(const TensorBase& fake, DispatchKey key) {
  auto& ctx = detail::getFakeTensorImpl(fake)->dispatch_context;

  if (auto pos = ctx.find(key); pos != ctx.end()) {
    return pos->second;
  } else {
    return nullptr;
  }
}

void* unsafeGetFakeContext(const TensorBase& fake, DispatchKey key) {
  auto& ctx = detail::getFakeTensorImpl(fake)->dispatch_context;

  if (auto pos = ctx.find(key); pos != ctx.end()) {
    return pos->second.get();
  } else {
    return nullptr;
  }
}

}  // namespace torchdistx
