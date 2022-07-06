# Owner(s): ["oncall: distributed"]
import copy
import sys
import tempfile
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.algorithms.model_averaging import averagers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchdistx.slow_momentum_fsdp import slowMomentum_hook, slowMomentum_optimizer

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class Net(nn.Module):

    def __init__(self, has_wrapping, sharding_strategy):
        # to ensure determinism
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        super().__init__()

        if has_wrapping:
            self.net = FSDP(nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                FSDP(
                    nn.Linear(16, 8),
                    device_id=torch.cuda.current_device(),
                    sharding_strategy=sharding_strategy
                )
            ),
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            )

        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(F.relu(self.net(x)))


class TestCommunicationHooks(FSDPTest):

    def _init_one_layer_fsdp(self, sharding_strategy):
        net = torch.nn.Linear(1, 5, bias=False)
        return FSDP(
            net,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_hook_with_grad_sync(
        self,
        sharding_strategy: Optional[ShardingStrategy],
    ):

        fsdp_net = self._init_one_layer_fsdp(sharding_strategy)
        input = torch.tensor([self.rank]).float().to(self.rank)

        slowMoState = slowMomentum_hook.SlowMoState(subgroup=None, grad_sync=True)
        # check that a default subgroup was created,
        # for small scale experiments equal to `World_Size`
        self.assertEqual(slowMoState.subgroup.size(), dist.get_world_size())

        cur_subgroup = dist.new_group(ranks=[self.rank])
        self.assertEqual(cur_subgroup.size(), 1)
        slowMoState = slowMomentum_hook.SlowMoState(cur_subgroup, grad_sync=True)
        # check that state has subgroup registered
        self.assertEqual(slowMoState.subgroup.size(), cur_subgroup.size())
        self.assertEqual(slowMoState.subgroup.rank(), 0)

        fsdp_net.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)

        fsdp_net.zero_grad()
        loss = fsdp_net(input).sum()
        loss.backward()

        # Make sure grads were not reduced, since each subgroup is only one worker.
        # Gradient in this case is equal to rank
        self.assertEqual(fsdp_net.params[0].grad[0], self.rank)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_hook_no_grad_sync(
        self,
        sharding_strategy: Optional[ShardingStrategy],
    ):

        fsdp_net = self._init_one_layer_fsdp(sharding_strategy)
        input = torch.tensor([self.rank]).float().to(self.rank)

        # create a subgroup equal to the whole WORLD
        cur_subgroup = dist.distributed_c10d._get_default_group()
        self.assertEqual(cur_subgroup.size(), dist.get_world_size())
        slowMoState = slowMomentum_hook.SlowMoState(cur_subgroup, grad_sync=False)
        # check that state has subgroup registered
        self.assertEqual(slowMoState.subgroup.size(), cur_subgroup.size())

        fsdp_net.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)
        fsdp_net.zero_grad()
        loss = fsdp_net(input).sum()
        loss.backward()

        # Make sure grads were not reduced, since `grad_sync` is set to False
        # Gradient in this case is equal to rank
        self.assertEqual(fsdp_net.params[0].grad[0], self.rank)

    def _train_step(self, inpt, net, optim):
        optim.zero_grad()
        loss = net(inpt).sum()
        loss.backward()
        optim.step()

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_averager(
        self,
        sharding_strategy: Optional[ShardingStrategy]
    ):
        fsdp_net = FSDP(
            Net(has_wrapping=True, sharding_strategy=sharding_strategy),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)
        fsdp_net_slowmo = FSDP(
            Net(has_wrapping=True, sharding_strategy=sharding_strategy),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)

        cur_subgroup = dist.new_group(ranks=[self.rank])
        slowMoState = slowMomentum_hook.SlowMoState(cur_subgroup)
        fsdp_net.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)
        fsdp_net_slowmo.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)
        inpt = torch.randn(7, 8).float().to(self.rank)

        slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=6,
            slowmo_factor=0.5,
            slowmo_lr=0.1,
        )

        # Manually changing slow momentum optimizer's averager's period
        # to differ from `slowmo_freq` to check it independently from
        # the momentum's update. Basically, parameter averaging now will happen
        # every 3rd step and momentum step every 6th.
        slowmo_optim.averager.period = 3

        averager2 = averagers.PeriodicModelAverager(
            period=3,
            process_group=dist.distributed_c10d._get_default_group()
        )
        base_optimizer = torch.optim.SGD(fsdp_net.parameters(), lr=1e-2)

        for _ in range(4):
            self._train_step(inpt, fsdp_net, base_optimizer)
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)
            averager2.average_parameters(list(fsdp_net.parameters()))

        for slowmo_params, net_params in zip(fsdp_net_slowmo.parameters(), fsdp_net.parameters()):
            self.assertEqual(slowmo_params, net_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_momentum_step(
        self,
        sharding_strategy: Optional[ShardingStrategy]
    ):
        net = torch.nn.Linear(2, 1)

        fsdp_net = FSDP(
            copy.deepcopy(net),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)
        fsdp_net_slowmo = FSDP(
            copy.deepcopy(net),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)

        cur_subgroup = dist.new_group(ranks=[self.rank])
        slowMoState = slowMomentum_hook.SlowMoState(cur_subgroup)
        fsdp_net.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)
        fsdp_net_slowmo.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)
        inpt = torch.tensor([(self.rank + 1)] * 2).float().to(self.rank)

        slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=1,
            slowmo_factor=0.5,
            slowmo_lr=0.1,
        )

        averager2 = averagers.PeriodicModelAverager(
            period=1,
            process_group=dist.distributed_c10d._get_default_group()
        )
        base_optimizer = torch.optim.SGD(fsdp_net.parameters(), lr=1e-2)

        for k, v in slowmo_optim.state.items():
            initial_prev_params = copy.deepcopy(v["prev_param"])
            initial_slow_momentum_buffer = copy.deepcopy(v["slow_momentum"])

        for _ in range(1):
            self._train_step(inpt, fsdp_net, base_optimizer)
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)
            averager2.average_parameters(list(fsdp_net.parameters()))

        # parameters before slow momentum update and after averaging
        # are in `fsdp_net.params[0]`
        # can use them to calculate momentum update
        # momentum_(t+1) = slowmo_factor * momentum_t +
        #   (prev_param - cur_param)/base_lr
        momentum = slowmo_optim.slowmo_factor * initial_slow_momentum_buffer\
            + (initial_prev_params - fsdp_net.params[0].data) / 0.01

        # parameter_(t+1) = prev_param - slowmo_lr * base_lr * momentum_(t+1)
        calculated_params = initial_prev_params - 0.1 * 0.01 * momentum

        self.assertEqual(fsdp_net_slowmo.params[0].data, calculated_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_state_dict(
        self,
        sharding_strategy: Optional[ShardingStrategy]
    ):
        chkpt = tempfile.gettempdir() + "/checkpoint.pt"
        fsdp_net_slowmo = FSDP(
            Net(has_wrapping=False, sharding_strategy=sharding_strategy),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)

        cur_subgroup = dist.new_group(ranks=[self.rank])
        slowMoState = slowMomentum_hook.SlowMoState(cur_subgroup)
        fsdp_net_slowmo.register_comm_hook(slowMoState, slowMomentum_hook.slowMo_hook)
        inpt = torch.randn(7, 8).float().to(self.rank)

        slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=4,
            slowmo_factor=0.5,
            slowmo_lr=0.1,
        )

        for _ in range(10):
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)

        state = {'optim_state_dict': slowmo_optim.state_dict()}

        if self.rank == 0:
            torch.save(state, chkpt)

        dist.barrier()

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(chkpt, map_location=map_location)

        slowmo_optim_dummy = slowMomentum_optimizer.SlowMomentumOptimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=2,
            slowmo_factor=3,
            slowmo_lr=4,
        )
        slowmo_optim_dummy.load_state_dict(checkpoint['optim_state_dict'])
        # make sure acerager's period and step were updated
        self.assertEqual(
            slowmo_optim_dummy.averager.period,
            slowmo_optim.averager.period
        )
        self.assertEqual(
            slowmo_optim_dummy.averager.step,
            slowmo_optim.averager.step
        )

        # make sure slowmo parameters were updated
        self.assertEqual(
            slowmo_optim_dummy.slowmo_freq,
            slowmo_optim.slowmo_freq
        )
        self.assertEqual(
            slowmo_optim_dummy.slowmo_factor,
            slowmo_optim.slowmo_factor
        )
        self.assertEqual(
            slowmo_optim_dummy.slowmo_lr,
            slowmo_optim.slowmo_lr
        )

        for _ in range(10):
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim_dummy)

        self.assertEqual(
            slowmo_optim_dummy.averager.step, 20
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_warnings(
        self,
        sharding_strategy: Optional[ShardingStrategy]
    ):
        net = torch.nn.Linear(1, 3, bias=False)
        with self.assertRaisesRegex(ValueError, "Base optimizer is a required parameter."):
            slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
                base_optim=None,
                slowmo_freq=4,
                slowmo_factor=0.5,
                slowmo_lr=0.1,
            )

        with self.assertRaisesRegex(ValueError, "Invalid ``slowmo_freq`` parameter, must be a positive value."):
            slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
                base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
                slowmo_freq=-3,
                slowmo_factor=0.5,
                slowmo_lr=0.1,
            )

        with self.assertRaisesRegex(ValueError, "Invalid ``slowmo_factor`` parameter, must be non-negative."):
            slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
                base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
                slowmo_freq=4,
                slowmo_factor=-0.5,
                slowmo_lr=0.1,
            )

        with self.assertRaisesRegex(ValueError, "Invalid ``slowmo_lr`` parameter, must be non-negative."):
            slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
                base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
                slowmo_freq=4,
                slowmo_factor=0.5,
                slowmo_lr=-0.1,
            )
        # check buffers and prev_params were initiated
        slowmo_optim = slowMomentum_optimizer.SlowMomentumOptimizer(
            base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
            slowmo_freq=4,
            slowmo_factor=0.5,
            slowmo_lr=0.1,
        )

        for _, v in slowmo_optim.state.items():
            self.assertEqual(
                v["slow_momentum"],
                torch.zeros(size=(3, 1), device=torch.cuda.current_device())
            )
            self.assertEqual(
                v["prev_param"],
                net.weight
            )


instantiate_parametrized_tests(TestCommunicationHooks)

if __name__ == "__main__":
    run_tests()
