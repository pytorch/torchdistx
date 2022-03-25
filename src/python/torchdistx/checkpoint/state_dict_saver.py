import io
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
    ShardMetadata,
)

from .metadata import (
    Metadata,
    BytesWriteRequest,
    ExtendedTensorMetadata,
    StorageMetadata,
    TensorWriteRequest,
)
from .storage_writer import StorageWriter

# -------------- private functions --------------
def _compute_tensor_md(fqn: str, tensor: Tensor, metadata: Metadata) -> None:
    # --- Step 3, populate the metadata ---
    #
    # Since torch.Tensor does not have a standard set of metadata we can operate on
    # We wrap troch.Tensor's metadata with ShardMetadata
    # This is frankly a bad idea, I will need to change this
    tensor = tensor.detach()
    tensor_size = list(tensor.size())
    storage_size = tensor.nelement() * tensor.element_size()
    shard_metadata = ShardMetadata(
        shard_offsets=[0] * len(tensor_size),
        shard_sizes=tensor_size,
        # Not sure how to deal with placement for regular tensor yet.
        # Since they are only keep the copy on rank0, let's hard code it for now.
        placement=f"rank:0/{str(tensor.device)}",
    )

    stm = ShardedTensorMetadata(
        shards_metadata=[shard_metadata],
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )

    etmd = ExtendedTensorMetadata(
        tensor_metadata=stm,
        storage_metadata=[
            StorageMetadata(
                shard_metadata=shard_metadata,
                storage_key=fqn,
                length=storage_size,
                offset=0,
            )
        ],
    )
    metadata.state_dict_metadata[fqn] = etmd


def _compute_sharded_tensor_md(
    fqn: str, tensor: ShardedTensor, metadata: Metadata
) -> None:
    smd = []
    for shard_md in tensor.metadata().shards_metadata:
        # each shard is in it own storage key.
        # Most network file system is optimized with single write, multiple read
        # Unless we can group tensors locally into one big chunk
        # It might be best to write each shard as one key
        suffix = "_".join([str(i) for i in shard_md.shard_offsets])
        storage_key = f"{fqn}_{suffix}"

        shard_size = 1
        for d in shard_md.shard_sizes:
            shard_size *= d

        # not particularly great
        storage_size = shard_size * tensor.local_shards()[0].tensor.element_size()

        one_smd = StorageMetadata(
            shard_metadata=shard_md,
            storage_key=storage_key,
            length=storage_size,
            offset=0,
        )
        smd.append(one_smd)

    etmd = ExtendedTensorMetadata(
        tensor_metadata=tensor.metadata(),
        storage_metadata=smd,
    )
    metadata.state_dict_metadata[fqn] = etmd


def _populate_inplace_with_a_tensor(
    fqn: str,
    tensor: Tensor,
    size_for_storage_keys: Dict[str, int],
    write_requests: List[TensorWriteRequest],
) -> None:
    # --- Step 1, populate write request ---
    tensor = tensor.detach()
    storage_size = tensor.nelement() * tensor.element_size()

    wr = TensorWriteRequest(
        tensor=tensor,
        storage_key=fqn,
    )

    write_requests.append(wr)

    # --- Step 2, populate the size_for_storage_keys ---
    #
    size_for_storage_keys[fqn] = storage_size


def _populate_inplace_with_a_sharded_tensor(
    fqn: str,
    sharded_tensor: ShardedTensor,
    size_for_storage_keys: Dict[str, int],
    write_requests: List[TensorWriteRequest],
) -> None:
    for shard in sharded_tensor.local_shards():
        # each shard has its own storage key.
        # For most cases, the read is a recovery from a failure to the same sharding
        # and does not need any resharding, write each shard as is is the most effective
        suffix = "_".join([str(i) for i in shard.metadata.shard_offsets])
        storage_key = f"{fqn}_{suffix}"

        tensor = shard.tensor.detach()
        storage_size = tensor.nelement() * tensor.element_size()

        assert (
            storage_key not in size_for_storage_keys
        ), "storage key must be unique per state_dict!"
        size_for_storage_keys[storage_key] = storage_size

        wr = TensorWriteRequest(
            tensor=tensor,
            storage_key=storage_key,
        )
        write_requests.append(wr)


def _prepare(
    state_dict: Dict[str, Any]
) -> Tuple[Metadata, Dict[str, int], List[BytesWriteRequest], List[TensorWriteRequest]]:
    """
    Uses the state_dict to build three things.

    metadata: Metadata
        The metatdata discribing the tensor / sharded tensor.
        And it is storage meta data. See "../metadata.py" for detail

    size_for_storage_keys: Dict[str, int]
        Key is the storage key name, value is its size
        It can used to pre allocate the storage for parallel and non sequential writes.

    tensor_write_requests: List[TensorWriteRequest]
        List of tensor write requests that should br perfromed by the writer.

    bytes_write_requests: List[BytesWriteRequest]
        List of byte write requests that should br perfromed by the writer.

    Subclasses can optionally overwrite the implementation here,
    if the default does not meet its requirement.
    """
    metadata = Metadata(state_dict_metadata={})
    storage_keys: Dict[str, int] = {}
    tensor_write_requests: List[TensorWriteRequest] = []
    bytes_write_requests: List[BytesWriteRequest] = []

    for fqn, obj in state_dict.items():
        if isinstance(obj, Tensor):
            # The assumption is that non ShardedTensors are full replicated across all ranks
            # So we just need one from Rank 0.
            # If that's not the case, we will update later.
            if dist.is_initialized() and dist.get_rank() != 0:
                pass
            else:
                _populate_inplace_with_a_tensor(
                    fqn, obj, storage_keys, tensor_write_requests
                )
                _compute_tensor_md(fqn, obj, metadata)
        elif isinstance(obj, ShardedTensor):
            _populate_inplace_with_a_sharded_tensor(
                fqn, obj, storage_keys, tensor_write_requests
            )
            _compute_sharded_tensor_md(fqn, obj, metadata)

        else:
            bytes_io = io.BytesIO()
            torch.save(obj, bytes_io)
            bwr = BytesWriteRequest(
                bytes=bytes_io,
                storage_key=fqn,
            )
            bytes_write_requests.append(bwr)

    return (metadata, storage_keys, bytes_write_requests, tensor_write_requests)


def save_state_dict(
    state_dict: Dict[str, Any],
    storage_writer: StorageWriter,
) -> None:
    """
    This public function defined the default behavior to save a state_dict
    Notes
    1. This is a WIP, the state_dict save with different versions of the code might not be compatible.
    2. The caller needs to ensure the correctness of the state_dict

    Sample Code
    ```
        my_model = MyModule()
        optimizer = Adagrad(my_model.parameters())
        ...

        model_state_dict = my_model.state_dict()
        optim_state_dict = optimizer.state_dict()

        ...
        # torchdistx.checkpoint does not assume the the correctness of the state_dict
        # the caller needs to ensure the correctness of the state_dict
        optim_state_dict = some_function_to_cleanup_optim_state_dict(optim_state_dict)
        ...

        fs_storage_writer = torchdistx.checkpoint.FileSystemWriter("/checkpoint/1")
        torchdistx.checkpoint.save_state_dict(
            state_dict=model_state_dict,
            storage_writer=fs_stroage_writer,
        )
        torchdistx.checkpoint.save_state_dict(
            state_dict=optim_state_dict,
            storage_writer=fs_stroage_writer,
        )
        ...
    ```

    Args:
        state_dict (Dict[str, Any]) : A state_dict
        storage_writer (StorageWriter): An instance of storage writer that performance the writes.
    """
    (
        metadata,
        storage_keys,
        bytes_write_requests,
        tensor_write_requests,
    ) = _prepare(state_dict)
    storage_writer.prepare_storage(storage_keys=storage_keys)
    storage_writer.write_metadata(metadata=metadata)
    bytes_futures = storage_writer.write_bytes(bytes_write_requests)
    tensor_futures = storage_writer.write_tensors(tensor_write_requests)
    bytes_futures.wait()
    tensor_futures.wait()
