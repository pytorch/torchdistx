import abc
import io
import os
import pickle
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.futures import Future

from .metadata import Metadata, BytesWriteRequest, TensorWriteRequest


class StorageWriter(abc.ABC):
    """
    Interface to write to underlying storage system
    """

    @abc.abstractmethod
    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        """
        Performs a write request and returns a Future to wait on.
        Args:
            requests (BytesWriteRequest): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        """
        Performs a write request and returns a Future to wait on.
        Args:
            requests (TensorWriteRequest): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def write_metadata(self, metadata: Metadata) -> None:
        """
        Writes the metatdata.

        Args:
            metadata (Metadata): see `./metadata.py`
        """
        pass

    def prepare_storage(self, storage_keys: Dict[str, int]) -> None:
        """
        This blocking call can be overwritten by the subclass.
        It can use `storage_keys` to plan for any write preformace optimization.
        e.g. non sequential and parallel writes.
        By default, it does nothing

        Args:
            storage_keys (Dict[str, int]): key - handle's name. value - size
                of the handle.
        """
        pass


class FileSystemWriter(StorageWriter):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def write_bytes(self, requests: List[BytesWriteRequest]) -> Future[None]:
        for req in requests:
            with open(os.path.join(self.path, req.storage_key), "wb") as storage:
                storage.write(req.bytes.getbuffer())

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    def write_tensors(self, requests: List[TensorWriteRequest]) -> Future[None]:
        for req in requests:
            with open(os.path.join(self.path, req.storage_key), "wb") as storage:
                # The following couple lines are simple implementation to get things going.
                #
                # At load time, to enable resharding, we use (sub)view of the tensor.
                # Since the storage of the tensor might not be contiguous. we need to
                # preseve the original view, to calculate the correct sub view at load.
                #
                # `torch.save` saves both the view and storage, it is a good option for unblocking
                # There are two drawbacks
                # 1. `torch.save` is pickle based, and pickle is not known for its compatibility,
                #    we should consider replacing it with a more stable option.
                # 2. pickle is not streamable.
                buffer = io.BytesIO()
                torch.save(req.tensor, buffer)
                storage.write(buffer.getbuffer())

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in Storage Writer
    def write_metadata(self, metadata: Metadata) -> None:
        # Only need to write the metadata once, since each ShardMetadata has the global view
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        with open(os.path.join(self.path, ".metadata"), "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
