# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional

from torch import Tensor
from torch.nn import Module

# We import `fake` to monkey-patch `repr()` of `Tensor`.
from . import fake  # noqa: F401
from . import _C


def deferred_init(module_fn: Callable[..., Module], *args, **kwargs) -> Module:
    """Defers the initialization of a ``Module``.

    This function forces all tensors constructed within ``module_fn`` to be
    fake while also recording all operations performed on them. The modules
    and tensors returned from ``module_fn`` can later be instantiated using
    the :func:`materialize_tensor` and :func:`materialize_module` functions.

    Args:
        module_fn:
            A callable that takes arbitrary number of arguments and returns a
            ``Module`` instance.
        args, kwargs:
            The positional and keyword arguments to be passed to ``module_fn``.
    """
    _C.enable_deferred_init(True)
    try:
        return module_fn(*args, **kwargs)
    finally:
        _C.enable_deferred_init(False)


def materialize_tensor(tensor: Tensor) -> Tensor:
    """Materializes ``tensor``.

    Args:
        module:
            The tensor instance to materialize.
    """
    return _C.materialize_tensor(tensor)


def materialize_module(
    module: Module,
    buffers_only: bool = False,
    check_fn: Optional[Callable[[Module], bool]] = None,
) -> None:
    """Materializes ``module``.

    Args:
        module:
            The module instance to materialize.
        buffers_only:
            A boolean value indicating whether to materialize the buffer tensors
            only.
        check_fn:
            An optional callable which takes a ``Module`` instance and returns a
            boolean value indicating whether to materialize it.
    """

    def materialize_tensors(tensors: Dict[str, Optional[Tensor]]) -> None:
        for key, tensor in tensors.items():
            if tensor is None:
                continue

            try:
                tensors[key] = _C.materialize_tensor(tensor)
            except ValueError:
                raise ValueError(f"'{key}' has already been materialized.") from None

    # Materialize the child modules recursively.
    for m in module.children():
        materialize_module(m, buffers_only, check_fn)

    # Materialize this module, possibly based on a check.
    if check_fn is None or check_fn(module):
        if not buffers_only:
            materialize_tensors(module._parameters)  # type: ignore[arg-type]

        materialize_tensors(module._buffers)
