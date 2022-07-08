# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers


class SlowMomentumOptimizer(torch.optim.Optimizer):
    r"""
    Wraps an arbitrary :class:`torch.optim.Optimizer` and runs
    FSDP distributed training with
    `Slow Momentum <https://arxiv.org/abs/1910.00643>`_.
    Currently, only available for FSDP modules defined
    with a `NO_SHARD` strategy.

    Args:
        base_optim: The base optimizer, which updates local instance of a model
        slowmo_freq: Specifies how often (number of iterations) slow momentum
            is to be performed (default: 48)
        slowmo_factor: This specifies the value of slowmo momentum
            to be used (default: 0.5)
        slowmo_lr: This specifies the value of slowmo learning rate
            to be used (default: 1.0)

    Example::

        >>>  import torch
        >>>  import torch.distributed as dist
        >>>  from torch.distributed.fsdp import(
        >>>    FullyShardedDataParallel as FSDP
        >>>  )
        >>>  from torchdistx.slow_momentum import(
        >>>     slow_momentum_comm,
        >>>     slow_momentum_optimizer
        >>>  )
        >>>
        >>>  net = torch.nn.Linear(4,10)
        >>>  fsdp_net = FSDP(net)
        >>>
        >>>  # This implementation communicate gradients between
        >>>  # workers of the same node,
        >>>  # before averaging of model's parameters between nodes.
        >>>  # The following creates intra-node subgroups
        >>>  # and SlowMoState will take care of storing all required
        >>>  # parameters for intra-node communication,
        >>>  # i.e. pre- and post-division factors, and subgroups.
        >>>  # To disable any communication between workers,
        >>>  # set `grad_sync` to `False`
        >>>  cur_subgroup, _ = dist.new_subgroups()
        >>>  slowMoState = slow_momentum_comm.SlowMoState(
        >>>     cur_subgroup,
        >>>     grad_sync=True
        >>>  )
        >>>
        >>>  # Register SlowMo hook, which only communicates gradients
        >>>  # in a intra-node fashion.
        >>>  fsdp_net.register_comm_hook(
        >>>     slowMoState,
        >>>     slow_momentum_comm.slowmo_hook
        >>>  )
        >>>
        >>>  base_optimizer = torch.optim.SGD(
        >>>     fsdp_net_slowmo.parameters(),
        >>>     lr=1e-2
        >>>  )
        >>>  # Create a SlowMo optimizer that wraps a local optimizer.
        >>>  slowmo_optim = slow_momentum_optimizer.SlowMomentumOptimizer(
        >>>     base_optim=base_optimizer,
        >>>     slowmo_freq=6,
        >>>     slowmo_factor=0.5,
        >>>     slowmo_lr=0.1
        >>>  )
        >>>
        >>>  # SlowMo runs intra-node gradient averaging at every step,
        >>>  # every 6th step it will run model averaging and
        >>>  # a slow momentum update.
        >>>  for step in range(0, 200):
        >>>     slowmo_optim.zero_grad()
        >>>     loss = loss_fn(output, labels)
        >>>     loss.backward()
        >>>     slowmo_optim.step()
    """

    def __init__(
        self,
        base_optim: torch.optim.Optimizer,
        slowmo_freq: int = 48,
        slowmo_factor: float = 0.5,
        slowmo_lr: float = 1.0,
    ):
        if base_optim is None:
            raise ValueError("Base optimizer is a required parameter.")
        self._base_optim = base_optim

        # check that base optimizer's learning rate is stored in param_groups
        if not (
            self._base_optim.param_groups and self._base_optim.param_groups[0]["lr"]
        ):
            raise ValueError(
                "Provided base optimizer does not have "
                "parameters or learning rate specified."
            )
        self.param_groups = self._base_optim.param_groups
        self.base_lr = self.param_groups[0]["lr"]

        if slowmo_freq < 1:
            raise ValueError(
                "Invalid ``slowmo_freq`` parameter, must be a positive value."
            )
        self.slowmo_freq = slowmo_freq

        if slowmo_factor < 0.0:
            raise ValueError(
                "Invalid ``slowmo_factor`` parameter, must be non-negative."
            )
        self.slowmo_factor = slowmo_factor

        if slowmo_lr < 0.0:
            raise ValueError("Invalid ``slowmo_lr`` parameter, must be non-negative.")
        self.slowmo_lr = slowmo_lr

        self.averager = averagers.PeriodicModelAverager(
            period=slowmo_freq, warmup_steps=0
        )
        self._init_slowmo_buffer()

    def _init_slowmo_buffer(self):
        for group in self.param_groups:
            for param in group["params"]:
                # Initialize momentums and memorize initial parameters
                self.state[param] = {
                    "slow_momentum": torch.zeros(
                        param.data.shape, device=torch.cuda.current_device()
                    ),
                    "prev_param": param.data.detach().clone(),
                }

    @property
    def state(self):
        return self._base_optim.state

    def __repr__(self):
        return self._base_optim.__repr__()

    def state_dict(self):
        r"""
        This is the same as :class:`torch.optim.Optimizer`
        :meth:`state_dict`, but adds an extra entries to record
        Slow Momentum's specific parameters: `slowmo_freq`,
        `slowmo_factor`, `slowmo_lr`, and `step` for the model's averager.
        """
        optim_state_dict = self._base_optim.state_dict()
        optim_state_dict["slowmo_freq"] = self.slowmo_freq
        optim_state_dict["slowmo_factor"] = self.slowmo_factor
        optim_state_dict["slowmo_lr"] = self.slowmo_lr
        optim_state_dict["step"] = self.averager.step

        return optim_state_dict

    def load_state_dict(self, state_dict):
        r"""
        This is the same as :class:`torch.optim.Optimizer`
        :meth:`load_state_dict`, but also restores Slow Momentum's
        specific parameters, saved in the provided ``state_dict``.
        Additionally, it restors `base_lr`, which refers to
        the base optimizer's learning rate, and re-initializes slow momentum
        buffers.
        """
        self._base_optim.load_state_dict(state_dict)
        self.slowmo_freq = state_dict["slowmo_freq"]
        self.slowmo_factor = state_dict["slowmo_factor"]
        self.slowmo_lr = state_dict["slowmo_lr"]
        self.averager.period = state_dict["slowmo_freq"]
        self.averager.step = state_dict["step"]
        # check that base optimizer's learning rate is stored in param_groups
        if not (self.param_groups and self.param_groups[0]["lr"]):
            raise ValueError(
                "Base optimizer does not have parameters or learning rate specified."
            )
        self.base_lr = self.param_groups[0]["lr"]
        self._init_slowmo_buffer()

    def step(self):
        r"""
        Performs a single optimization step (parameter update)
        and a slow momentum update. Slow momentum update involves
        model's exact averaging of parameters and a momentum update,
        which happens every `slowmo_freq` step.
        """
        self._base_optim.step()
        self.averager.average_parameters(params=self.param_groups)
        if self.averager.step % self.slowmo_freq == 0:
            for group in self.param_groups:
                for param in group["params"]:
                    # Update the slow momentum
                    p_state = self.state[param]

                    p_state["slow_momentum"].mul_(self.slowmo_factor).sub_(
                        param.data, alpha=1 / self.base_lr
                    ).add_(p_state["prev_param"], alpha=1 / self.base_lr)

                    # Update parameters
                    p_state["prev_param"].add_(
                        p_state["slow_momentum"], alpha=-self.slowmo_lr * self.base_lr
                    )
                    param.data.copy_(p_state["prev_param"])

    def zero_grad(self, set_to_none: bool = False):  # type: ignore[override]
        self._base_optim.zero_grad(set_to_none=set_to_none)
