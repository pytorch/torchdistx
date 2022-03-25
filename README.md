# torchdistX - PyTorch Distributed Experimental

The torchdistX package contains a collection of experimental features for
PyTorch Distributed. Its purpose is to gather feedback from our users before
introducing those features in the core PyTorch package. In a sense features
included in torchdistX can be considered in an incubation period. As its name
suggests, this package should not be used in production and its content is
always subject to change.

As of today the following features are available in torchdistX:

- Fake Tensor (TODO: add doc link)
- Deferred Module Initialization (TODO: add doc link)

## Dependencies
torchdistX versions corresponding to each PyTorch release:

| `torch`      | `torchdistx` | `python`         |
| ------------ | ------------ | ---------------- |
| `master`     | `main`       | `>=3.7`, `<=3.9` |
| `1.11.0`     | `0.11.0`     | `>=3.7`, `<=3.9` |

## Installation
Note that only Linux and macOS operating systems are supported. There are no
plans to introduce Windows support.

### Conda
Conda is the recommended way to install torchdistX. Running the following
command in a Conda environment will install torchdistX and all its dependencies.

```
conda install -c <TODO: add channel> torchdistx
```

In fact torchdistX offers several Conda packages that you can install
independently based on your needs:

| Package                                                                    | Description                                      |
|----------------------------------------------------------------------------|--------------------------------------------------|
| [torchdistx](https://anaconda.org/torchdistx/torchdistx)                   | torchdistX Python Library                        |
| [torchdistx-cc](https://anaconda.org/torchdistx/torchdistx-cc)             | torchdistX C++ Runtime Library                   |
| [torchdistx-cc-devel](https://anaconda.org/torchdistx/torchdistx-cc-devel) | torchdistX C++ Runtime Library Development Files |
| [torchdistx-cc-debug](https://anaconda.org/torchdistx/torchdistx-cc-debug) | torchdistX C++ Runtime Library Debug Symbols     |

### PyPI
```
TBD
```

### From Source
#### Prerequisites
- The build process requires CMake 3.21 or later. If you are in a Conda
  environment, you can install an up-to-date version by executing
  `conda install -c conda-forge cmake`. For other environments, please refer to your package manager or [cmake.org](https://cmake.org/download/).
- After cloning the repository make sure to initialize all submodules by
  executing `git submodule update --init --recursive`.

Once you have all prerequisites run the following commands to install the
torchdistX Python package:

```
$ cmake -DTORCHDIST_INSTALL_STANDALONE=ON -B build
$ cmake --build build
$ pip install .
```

For advanced build options you can check out [CMakeLists.txt](./CMakeLists.txt).

#### Development
In case you would like to contribute to the project you can slightly modify the
commands listed above:

```
$ cmake -B build
$ cmake --build build
$ pip install -e .
```

With `pip install -e .` you enable the edit mode (a.k.a. develop mode) that
allows you to modify the Python files in-place without requiring to repeatedly
install the package. If you are working with the C++ code, whenever you modify a
header or implementation file, executing `cmake --build build` alone is
sufficient. You do not have to repeat the other commands.

The project also comes with [requirements.txt](./requirements.txt) if you would
like to fully set up a Python virtual environment.

#### Tip
Note that using the Ninja build system and ccache can significatly speed up your
build times. You can replace the initial CMake command listed above with the
following version to leverage them:

```
$ cmake -GNinja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build
```

## Getting Started

### Fake Tensor
In the example below we construct two fake tensors with the `fake_mode` context
manager. Fake tensors pretend to be real tensors, but they don't allocate any
storage. Internally they rely on the meta backend similar to meta tensors.

```python
>>> import torch
>>> from torchdistx import fake
>>>
>>> with fake.fake_mode():
...    a = torch.ones([10])
...    b = torch.ones([20], device="cuda")
...
>>> a
tensor(..., size=(10,), fake=True)
>>> b
tensor(..., size=(20,), device=cuda, fake=True)
```

### Deferred Module Initialization
This feature forces all tensors of a module to be constructed as fake while also
recording all operations performed on them. The module, its submodules, and its
tensors can later be materialized by calling the `materialize_module()` and
`materialize_tensor()` functions.

```python
>>> import torch
>>> from torchdistx import deferred_init
>>>
>>> m = deferred_init.deferred_init(torch.nn.Linear, 10, 20)
>>> m.weight
Parameter containing:
tensor(..., size=(20, 10), requires_grad=True, fake=True)
>>>
>>> deferred_init.materialize_module(m)
>>> m.weight
Parameter containing:
tensor([[-0.1838, -0.0080,  0.0747, -0.1663, -0.0936,  0.0587,  0.1988, -0.0977,
         -0.1433,  0.2620],
        ..., requires_grad=True)
```


### ShardedTensor checkpointing
This feature allows to perform SPMD checkpointing of state_dict featuring ShardedTensor.
It currently works by having each rank checkpointing their local shards and have rank `0`
deal with regular tensors, non-tensor items and metadata.

** WARNING ** This feature requires PyTorch master (or future 1.12) for ShardedTensor APIs.
** WARNING ** This feature depends on experimental PyTorch APIs.

```python
import torch
from torchdistx.checkpoint as cp
from torch.distributed._shard.sharded_tensor import state_dict_hook

def worker(rank):
  model = ...
  model._register_state_dict_hook(state_dict_hook)

  fs_writer = cp.FileSystemWriter(path="/checkpoint/")
  cp.save_state_dict(state_dict=model.state_dict(), storage_writer=fs_writer)
```


## Documentation
TBD

## Contributing
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).
