import os
import shutil
import tempfile
from typing import Dict, Optional, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)

from torchdistx.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)


def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: Dict[str, torch.Tensor],
    state_dict_2: Dict[str, torch.Tensor],
) -> bool:
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2), f"Key {key}'s tensor does not match"
            )
        elif isinstance(value_1, ShardedTensor):
            for local_shard_1, local_shard_2 in zip(
                value_1.local_shards(), value_2.local_shards()
            ):
                self.assertTrue(
                    torch.equal(local_shard_1.tensor, local_shard_1.tensor),
                    f"Key {key}'s shard does not match",
                )

    return True


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# The ShardedModels are borrowed from
# test/distributed/_sharded_tensor/test_sharded_tensor.py
class MyShardedModel3(torch.nn.Module):
    def __init__(
        self,
        spec: ShardingSpec,
    ) -> None:
        super(MyShardedModel3, self).__init__()
        self.sharded_tensor: ShardedTensor = sharded_tensor.empty(
            spec, 10, 20, init_rrefs=False
        )


class MyShardedModel2(torch.nn.Module):
    sharded_tensor2: Optional[ShardedTensor]

    def __init__(
        self,
        spec: Optional[ShardingSpec] = None,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super(MyShardedModel2, self).__init__()
        if spec is not None:
            self.sharded_tensor2 = sharded_tensor.empty(
                spec, 10, 20, process_group=group, init_rrefs=False
            )
        else:
            self.sharded_tensor2 = None
        self.random_tensor2 = torch.nn.Parameter(torch.rand(2, 2))


class MyShardedModel1(torch.nn.Module):
    sharded_tensor1: Optional[ShardedTensor]

    def __init__(
        self,
        spec: Optional[ShardingSpec] = None,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super(MyShardedModel1, self).__init__()
        if spec is not None:
            self.sharded_tensor1 = sharded_tensor.empty(
                spec, 10, 20, process_group=group, init_rrefs=False
            )
        else:
            self.sharded_tensor1 = None
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        self.submodule = MyShardedModel2(spec, group)


class TestStateDictSaveLoad(TestCase):
    def test_read_write_only_tensor(self) -> None:
        state_dict_to_save = MyTestModule().state_dict()
        path = tempfile.mkdtemp()

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # Genrate a new modle
        state_dict_to_load_to = MyTestModule().state_dict()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Load from file without any resharding
        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_read_write_shard_tenosr(self) -> None:
        paths = [tempfile.mkdtemp()]
        dist.broadcast_object_list(paths)

        path = paths[0]

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        model_to_save = MyShardedModel1(spec)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)  # noqa: E501
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()


class TestReshardOnLoad(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def get_file_path(self) -> str:
        paths = [tempfile.mkdtemp()]
        dist.broadcast_object_list(paths)
        return paths[0]

    def load_tensor(self, tensor: ShardedTensor) -> torch.Tensor:
        res = torch.zeros(tensor.shape, device="cpu") if dist.get_rank() == 0 else None
        tensor.gather(out=res)
        return cast(Tensor, res)

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_load_with_different_shard_plan(self) -> None:
        path = self.get_file_path()

        specs = [
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                ],
            ),
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:1/cuda:1",
                    "rank:0/cuda:0",
                ],
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0/cuda:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[2, 0],
                        shard_sizes=[1, 20],
                        placement="rank:1/cuda:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[3, 0],
                        shard_sizes=[3, 20],
                        placement="rank:0/cuda:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[6, 0],
                        shard_sizes=[3, 20],
                        placement="rank:1/cuda:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[9, 0],
                        shard_sizes=[1, 20],
                        placement="rank:0/cuda:0",
                    ),
                ]
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[8, 20],
                        placement="rank:1/cuda:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0/cuda:0",
                    ),
                ]
            ),
        ]

        for s0 in specs:
            for s1 in specs:
                if s0 == s1:
                    continue

                shutil.rmtree(path, ignore_errors=True)
                os.makedirs(path)
                dist.barrier()

                model_to_save = MyShardedModel3(s0)
                model_to_save._register_state_dict_hook(state_dict_hook)
                state_dict_to_save = model_to_save.state_dict()

                fs_writer = FileSystemWriter(path=path)
                save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

                dist.barrier()

                model_to_load = MyShardedModel3(s1)
                model_to_load._register_state_dict_hook(state_dict_hook)
                state_dict_to_load_to = model_to_load.state_dict()

                fs_reader = FileSystemReader(path=path)
                load_state_dict(
                    state_dict=state_dict_to_load_to, storage_reader=fs_reader
                )

                dist.barrier()

                store_tensor = self.load_tensor(model_to_save.sharded_tensor)
                load_tensor = self.load_tensor(model_to_load.sharded_tensor)

                if dist.get_rank() == 0:

                    self.assertTrue(
                        torch.allclose(store_tensor, load_tensor), msg=f"{s0} vs {s1}"
                    )

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_load_rowwise_to_colwise(self) -> None:
        path = self.get_file_path()

        src_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        dst_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        dist.barrier()

        model_to_save = MyShardedModel3(src_spec)
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        model_to_load = MyShardedModel3(dst_spec)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        fs_reader = FileSystemReader(path=path)

        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        store_tensor = self.load_tensor(model_to_save.sharded_tensor)
        load_tensor = self.load_tensor(model_to_load.sharded_tensor)

        if dist.get_rank() == 0:
            self.assertTrue(torch.allclose(store_tensor, load_tensor))
