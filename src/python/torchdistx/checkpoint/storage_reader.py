import abc
import operator
import os
import pickle
from typing import List, Optional, cast

import torch
from torch import Tensor
from torch.futures import Future

from .metadata import BytesReadRequest, Metadata, TensorReadRequest


class StorageReader(abc.ABC):
    """
    Interface to read from the underlying storage system.
    """

    @abc.abstractmethod
    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        """
        Read request and returns a Future to wait on.
        Args:
            requests (List[BytesReadRequest]): see `./metadata.py`]
        """
        pass

    @abc.abstractmethod
    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Performs a read request and returns a Future to wait on.
        Args:
            requests (List[BytesReadRequest]): see `./metadata.py`
        """
        pass

    @abc.abstractmethod
    def read_metadata(self) -> Metadata:
        """
        Read the meta data and returns.
        """
        pass


class FileSystemReader(StorageReader):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def read_tensors(self, requests: List[TensorReadRequest]) -> Future[None]:
        """
        Very basic implementation that read from file system.
        """
        # Sort the the requests by storage key and try to reuse the loaded tensors
        requests.sort(key=operator.attrgetter("storage_key"))

        cached_storage_key = None
        view_cached: Optional[Tensor] = None

        for req in requests:
            if cached_storage_key != req.storage_key:
                with open(os.path.join(self.path, req.storage_key), "rb") as storage:
                    view_cached = cast(Tensor, torch.load(storage))
                    cached_storage_key = req.storage_key

            view_to_copy: Tensor = cast(Tensor, view_cached)
            # FileSystemWrite writes the tensor as is during save.
            # During load time, we will load the Tensor (with it orignal view)
            # narrow it along all dimemsions, and copy_ it to the
            # target tensor, which will be the same size.
            for dim, (start, length) in enumerate(zip(req.offsets, req.lengths)):
                view_to_copy = cast(Tensor, torch.narrow(view_to_copy, dim, start, length))

            assert (
                view_to_copy.size() == req.tensor.size()
            ), f"The {req.storage_key} src/dst size does not match."

            req.tensor.copy_(view_to_copy)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_bytes(self, requests: List[BytesReadRequest]) -> Future[None]:
        for req in requests:
            with open(os.path.join(self.path, req.storage_key), "rb") as storage:
                req.bytes.write(storage.read())

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementating the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        with open(os.path.join(self.path, ".metadata"), "rb") as metadata_file:
            return pickle.load(metadata_file)
