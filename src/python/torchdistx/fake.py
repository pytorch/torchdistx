# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Callable, Iterator

import torch

from . import _C


# Since the `repr()` method of `Tensor` is not extensible we monkey-patch it
# to support fake tensors.
def _patch_tensor_repr() -> Callable[[torch.Tensor], str]:
    tensor_repr = torch.Tensor.__repr__

    def patched_repr(tensor: torch.Tensor) -> str:
        if _C.is_fake(tensor):
            s = f"tensor(..., size={tuple(tensor.shape)}"

            if tensor.dtype != torch.get_default_dtype():
                s += f", dtype={tensor.dtype}"

            if tensor.device.type != "cpu":
                s += f", device={tensor.device})"

            if tensor.requires_grad:
                s += ", requires_grad=True"

            return s + ", fake=True)"
        else:
            return tensor_repr(tensor)

    return patched_repr


torch.Tensor.__repr__ = _patch_tensor_repr()  # type: ignore[assignment]


@contextmanager
def fake_mode() -> Iterator[None]:
    """Instantiates all tensors within its context as fake."""
    _C.enable_fake_mode(True)
    try:
        yield
    finally:
        _C.enable_fake_mode(False)


def is_fake(tensor: torch.Tensor) -> bool:
    """Indicates whether ``tensor`` is fake."""
    return _C.is_fake(tensor)
