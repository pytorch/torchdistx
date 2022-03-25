import io
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from torch.distributed._shard.sharding_spec import (
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    _check_shard_metadata_pair_overlap,
)

from .metadata import BytesReadRequest, TensorReadRequest, Metadata
from .storage_reader import StorageReader

# -------------- private functions --------------
def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ShardMetadata, current_shard: ShardMetadata
) -> List[Tuple[int, int, int, int]]:
    """
    Return the overlapping region of saved_shard and current_shard on each dimention.
    """
    narrows = []
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.shard_offsets,
            current_shard.shard_offsets,
            saved_shard.shard_sizes,
            current_shard.shard_sizes,
        )
    ):
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows


def _reshard_and_prepare_read_request(
    state_dict: Dict[str, Any], metadata_from_storage: Metadata
) -> Tuple[List[BytesReadRequest], List[TensorReadRequest]]:
    """
    Use the loaded metadata and the current state dict to map the saved tensors to current tensors

    NOTE:
    During the save,
    """
    tensor_read_requests = []
    bytes_read_requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach()
            storage_size = tensor.nelement() * tensor.element_size()

            rr = TensorReadRequest(
                tensor=tensor,
                storage_key=fqn,
                offsets=tuple([0] * len(tensor.size())),
                lengths=tensor.size(),
            )

            tensor_read_requests.append(rr)
        elif isinstance(obj, ShardedTensor):
            md = metadata_from_storage.state_dict_metadata[fqn]

            # this is a naive quadratic algo that can later be optimized by sorting metadata and the shards md
            # FIXME what sort of error handling should we do? Overlapping storage items? Missing data?
            for shard in obj.local_shards():
                # scan all mds looking for chunks
                for storage_md in md.storage_metadata:
                    shard_md_from_storage = storage_md.shard_metadata
                    tensor = shard.tensor.detach()
                    assert shard_md_from_storage is not None
                    # FIXME what does it mean for offset > 0? just add it to read request offset?
                    assert (
                        storage_md.offset == 0
                    ), "Storage at key {fqn} is saved with an offset, we cannot load this yet"

                    # do they overlap?
                    if not _check_shard_metadata_pair_overlap(
                        shard.metadata, shard_md_from_storage
                    ):
                        continue

                    storage_key = storage_md.storage_key

                    target_tensor = tensor
                    offsets = []
                    lengths = []
                    for (
                        dim,
                        offset_for_saved_tensor,
                        offset_for_current_tensor,
                        length,
                    ) in _shards_get_overlap_region_wrt_saved_tensor(
                        saved_shard=shard_md_from_storage, current_shard=shard.metadata
                    ):
                        # Note that we do NOT want to make any tensor copy.
                        # all operation must be view only
                        target_tensor = torch.narrow(
                            target_tensor, dim, offset_for_current_tensor, length
                        )
                        offsets.append(offset_for_saved_tensor)
                        lengths.append(length)

                    rr = TensorReadRequest(
                        tensor=target_tensor,
                        storage_key=storage_key,
                        offsets=tuple(offsets),
                        lengths=tuple(lengths),
                    )
                    tensor_read_requests.append(rr)
        else:
            # This is actually hard to handle correctly
            # If the value is not a tensor but any random obj,
            # we cannot just write whatever memory it points to inplace
            # the best we can to is to replace it with an object of the same type
            bytes_io = io.BytesIO()
            brr = BytesReadRequest(
                bytes=bytes_io,
                storage_key=fqn,
            )
            bytes_read_requests.append(brr)

    return (bytes_read_requests, tensor_read_requests)


def load_state_dict(
    state_dict: Dict[str, Any],
    storage_reader: StorageReader,
) -> None:
    """
    This public function defines the default behavior to load a state_dict

    Sample Code
    ```
        my_model = MyModule()
        optimizer = Adagrad(my_model.parameters())
        ...

        model_state_dict = my_model.state_dict()
        optim_state_dict = optimizer.state_dict()
        ...

        # torch.distributed does not assume the the correctness of the state_dict
        # the caller needs to ensure the correctness of the state_dict
        optim_state_dict = some_function_to_cleanup_optim_state_dict(optim_state_dict)
        ...

        fs_storage_loader = torch.distributed.FileSystemLoader("/checkpoint/1")
        torch.distributed.load_state_dict(
            state_dict=model_state_dict,
            storage_reader=fs_stroage_loader,
        )
        torch.distributed.load_state_dict(
            state_dict=optim_state_dict,
            storage_reader=fs_stroage_loader,
        )

        # module.load_state_dict() functon might have customized steps
        # to flush the state_dict, must call them to
        # ensure the correct behavior
        my_model.load_state_dict(model_state_dict)
        optim_state_dict.load_state_dict(optim_state_dict)
        ...
    ```
    Args:
        state_dict (Dict[str, Any]) : A state_dict to load to. Note that this
            state dict will updated in places.
        storage_reader (StorageReader): An instance of storage loader.
    """

    metadata = storage_reader.read_metadata()
    bytes_read_requests, tensor_read_requests = _reshard_and_prepare_read_request(
        state_dict=state_dict, metadata_from_storage=metadata
    )
    bytes_futures = storage_reader.read_bytes(bytes_read_requests)
    tensor_futures = storage_reader.read_tensors(tensor_read_requests)
    bytes_futures.wait()

    # Addtional steps are required to convert the bytes to its original type
    # Note that this is NOT inplace,
    # it creating a new object and replace what's in the state dict
    for req in bytes_read_requests:
        fqn = req.storage_key
        # Ensure the BytesIO is rewound
        req.bytes.seek(0)
        state_dict[fqn] = torch.load(req.bytes)

    tensor_futures.wait()
