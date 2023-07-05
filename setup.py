# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------

import glob
import os
import shutil
import subprocess
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "detrex", "layers", "csrc")

main_source = os.path.join(extensions_dir, "vision.cpp")
sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
    os.path.join(extensions_dir, "*.cu")
)

sources = [main_source] + sources

extension = CppExtension

extra_compile_args = {"cxx": []}
define_macros = []

extension = CUDAExtension
sources += source_cuda
define_macros += [("WITH_CUDA", None)]
extra_compile_args["nvcc"] = [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

sources = [os.path.join(extensions_dir, s) for s in sources]
include_dirs = [extensions_dir]

ext_modules = [
    extension(
        "detrex._C",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="detrex._C",
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension}
)


if __name__ == "__main__":

    setup(
        name="detrex",
        version="0.3.0",
        author="International Digital Economy Academy",
        url="https://github.com/rentainhe/detrex",
        description="IDEA open source toolbox for transformer-based instance recognition tasks",
        license=license,
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(
            exclude=(
                "configs",
                "tests",
            )
        ),
        package_data={"detrex.config": get_detrex_configs()},
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
