# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchdistx.deferred_init import materialize_tensor


def test_materialize_tensor_is_noop_for_real_tensors():
    a = torch.ones([10])

    e = materialize_tensor(a)

    assert a is e
