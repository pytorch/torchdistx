# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torchdistx.optimizers.anyprecision_optimizer import AnyPrecisionAdamW
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

# import


# if not dist.is_available():
#     print("Distributed not available, skipping tests", file=sys.stderr)
#     sys.exit(0)


class TestAnyPrecisionOptimizer(TestCase):
    def test_adam_equivalence(self):
        """
        Tests that AnyPrecisionAdamW is equivalent to AdamW when
        kahan summation and different dtypes for momentum, variance,
        and compensation buffer are turned off (i.e. all float32).
        """
        model = nn.Sequential(
            nn.Linear(5, 5), nn.Linear(5, 5), nn.Linear(5, 5)
        )

        model_clone = deepcopy(model)

        # Test non-default options
        betas = (0.8, 0.88)
        weight_decay = 0.03

        adam_opt = optim.AdamW(
            model_clone.parameters(),
            betas=betas,
            weight_decay=weight_decay
        )
        anyprecision_adam = AnyPrecisionAdamW(
            model.parameters(),
            variance_dtype=torch.float32,
            betas=betas,
            weight_decay=weight_decay
        )

        # Verify params are equal initially
        model_orig_params = [p.clone() for p in model.parameters()]
        for p1, p2 in zip(model_clone.parameters(), model_orig_params):
            self.assertEqual(p1, p2)


        inp = torch.randn(5, 5)

        for i in range(6):
            adam_opt.zero_grad()
            anyprecision_adam.zero_grad()
            inp = torch.randn(5, 5)
            model(inp).sum().backward()
            model_clone(inp).sum().backward()
            adam_opt.step()
            anyprecision_adam.step()

            # Ensure params are modified from original
            if i == 0:
                for p1, p2 in zip(model.parameters(), model_orig_params):
                    self.assertNotEqual(p1, p2)

            for p1, p2 in zip(model.parameters(), model_clone.parameters()):
                self.assertEqual(p1, p2)



if __name__ == "__main__":
    run_tests()
