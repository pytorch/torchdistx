# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import List, TextIO

import torch
from setuptools import Command, setup
from setuptools.command.install import install as install_base
from setuptools.dist import Distribution as DistributionBase
from setuptools.errors import FileError  # type: ignore[attr-defined]

package_path = "src/python"

package_name = "torchdistx"


class Distribution(DistributionBase):
    # Since we are injecting our Python C extension into the package instead
    # of building it we need to mark the package as non-pure.
    def has_ext_modules(self) -> bool:
        return True


class install(install_base):
    install_base.sub_commands.append(("install_cmake", lambda self: True))

    def finalize_options(self) -> None:
        install_base.finalize_options(self)

        # Older versions of distutils incorrectly check `ext_modules` to
        # determine whether a package is non-pure. We override it here.
        if self.distribution.has_ext_modules():  # type: ignore[attr-defined]
            self.install_lib = self.install_platlib


# We inject our Python C extension and optionally our shared library into the
# package by installing them directly via CMake.
class install_cmake(Command):
    description = "install CMake artifacts"

    user_options = [
        ("cmake-build-dir=", "b", "build directory (where to install from)"),
        ("install-dir=", "d", "directory to install to"),
        ("standalone", "s", "bundle C++ library"),
        ("no-standalone", None, "don't bundle C++ library"),
    ]

    boolean_options = ["standalone"]

    negative_opt = {"no-standalone": "standalone"}

    def initialize_options(self) -> None:
        # This is a required option and specifies the build (a.k.a. binary)
        # directory of the CMake project to install.
        self.cmake_build_dir = "build"

        # If not specified, the value of this option is copied over from the
        # parent `install` command. It specifies the directory into which to
        # install the CMake artifacts.
        self.install_dir: str = None  # type: ignore[assignment]

        # By default we install a non-standalone package containing only the
        # Python C extension. For a wheel package this option must be set to
        # true to ensure that it also contains the shared library.
        self.standalone: bool = None  # type: ignore[assignment]

    def finalize_options(self) -> None:
        self.ensure_dirname("cmake_build_dir")

        # If not specified, copy the value of `install_dir` from the `install`
        # command.
        self.set_undefined_options("install", ("install_lib", "install_dir"))

        # If not specified, we infer the value of `standalone` from the CMake
        # configuration file.
        if self.standalone is None:
            self.standalone = self._should_install_standalone()

    def _should_install_standalone(self) -> bool:
        try:
            f = open(os.path.join(self.cmake_build_dir, "CMakeCache.txt"))
        except FileNotFoundError:
            raise FileError("CMakeCache.txt not found. Run CMake first.")

        # Parse the value of the `TORCHDIST_INSTALL_STANDALONE` option from the
        # CMake configuration file.
        with f:
            for line in f:
                if line.startswith("TORCHDIST_INSTALL_STANDALONE"):
                    _, value = line.strip().split("=", 1)

                    return value.upper() in ["1", "ON", "TRUE", "YES", "Y"]

        return False

    def run(self) -> None:
        # If the user has requested a standalone package, install the shared
        # library and other related artifacts into the package.
        if self.standalone:
            self._cmake_install()

        # Install the Python C extension.
        self._cmake_install(component="python")

    def _cmake_install(self, component: str = None) -> None:
        prefix_dir = os.path.join(self.install_dir, package_name)

        cmd = ["cmake", "--install", self.cmake_build_dir, "--prefix", prefix_dir]

        if self.verbose:  # type: ignore[attr-defined]
            cmd += ["--verbose"]

        if component:
            cmd += ["--component", component]

        # Ensure that we remove debug symbols from all DSOs.
        cmd += ["--strip"]

        # Run `cmake --install` in a subprocess.
        self.spawn(cmd)

    def get_inputs(self) -> List[str]:
        # We don't take any input files from other commands.
        return []

    def get_outputs(self) -> List[str]:
        # Since we don't have an easy way to infer the list of files installed
        # by CMake we don't support the `record` option.
        warnings.warn("`install_cmake` does not support recording output files.")

        return []


def open_file(*paths: str, write: bool = False) -> TextIO:
    file_path = os.path.join(os.path.dirname(__file__), *paths)

    return open(file_path, mode="w" if write else "r")


def read_package_version() -> str:
    with open_file("VERSION") as f:
        version = f.readline().strip()

    if os.getenv("TORCHDIST_SERVER_BUILD", "") != "":
        return version + os.getenv("TORCHDIST_PEP440_PRE_RELEASE", "")
    else:
        return version + ".dev0"


def write_version_file(version: str) -> None:
    with open_file(package_path, package_name, "version.py", write=True) as f:
        f.write(f'__version__ = "{version}"\n')


def read_long_description() -> str:
    with open_file("README.md") as f:
        return f.read()


def main() -> None:
    version = read_package_version()

    write_version_file(version)

    setup(
        distclass=Distribution,
        cmdclass={
            "install": install,  # type: ignore[dict-item]
            "install_cmake": install_cmake,
        },
        name="torchdistx",
        version=version,
        description="A collection of experimental features for PyTorch Distributed",
        long_description=read_long_description(),
        long_description_content_type="text/markdown",
        author="PyTorch Distributed Team",
        url="https://github.com/pytorch/torchdistx",
        license="BSD",
        keywords=["pytorch", "machine learning"],
        packages=[package_name],
        package_dir={"": package_path},
        package_data={"": ["*.pyi"]},
        python_requires=">=3.7",
        # Since PyTorch does not offer ABI compatibility we have to make sure
        # that we use the same version that was used at build time.
        install_requires=[f"torch=={torch.__version__}"],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
