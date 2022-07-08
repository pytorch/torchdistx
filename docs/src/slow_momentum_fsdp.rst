Slow Momentum for Fully Sharded Data Parallel training with ``NO_SHARD`` strategy
===================================================================================
Slow Momentum is a general framework to improve the accuracy of
communication-efficient distributed training methods. Slow Momentum Algorithm
requires exact-averaging of parameters before a momentum update, which is not feasible
in a scenario with sharded model's parameters. As a result current implementation
available only for FSDP ``NO_SHARD`` strategy.

API
---

The API consists of ``SlowMoState``, ``slowmo_hook``, and ``SlowMomentumOptimizer``.

.. autoclass:: torchdistx.slow_momentum.slow_momentum_comm.SlowMoState

.. autofunction:: torchdistx.slow_momentum.slow_momentum_comm.slowmo_hook

.. autoclass:: torchdistx.slow_momentum.slow_momentum_optimizer.SlowMomentumOptimizer
