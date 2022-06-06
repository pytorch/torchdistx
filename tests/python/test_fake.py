# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchdistx.fake import fake_mode, meta_like


def test_meta_like_works_as_expected():
    with fake_mode():
        a = torch.ones([10])

    b = meta_like(a)

    assert b.device.type == "meta"
    assert b.dtype == a.dtype
    assert b.size() == a.size()
    assert b.stride() == a.stride()


def test_meta_like_fails_if_not_fake():
    a = torch.ones([10])

    with pytest.raises(ValueError):
        meta_like(a)
