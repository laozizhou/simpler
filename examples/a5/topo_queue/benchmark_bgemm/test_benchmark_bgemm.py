#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark BGEMM on A5 with the topo_queue runtime.

Same workload as the host_build_graph copy; the only intended difference is the
runtime tag, so a result mismatch between the two is a topo_queue scheduling
bug by construction."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


@scene_test(level=2, runtime="topo_queue")
class TestBenchmarkBgemmTopoQueue(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/bgemm_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "GEMM",
                "source": "kernels/aic/kernel_gemm_tile.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "kernels/aiv/kernel_tile_add.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    # Scale0..Scale3 vary only the task count, so a sweep over them separates a
    # per-task cost that is paid in series from one the cores absorb in
    # parallel. They are manual on every platform: the sweep is a measurement,
    # not a correctness case, and Case0 already covers the shape.
    CASES = [
        {
            "name": "Case0",
            "platforms": ["a5sim", "a5"],
            "manual": ["a5sim"],
            "params": {"matmul_add_task_num": 500, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
        *(
            {
                "name": f"Scale{i}",
                "platforms": ["a5sim", "a5"],
                "manual": ["a5sim", "a5"],
                "params": {
                    "matmul_add_task_num": n,
                    "incore_data_size": 128,
                    "incore_loop": 4,
                    "grid_k": 2,
                },
            }
            for i, n in enumerate((100, 250, 1000, 2000))
        ),
    ]

    def generate_args(self, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        B = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        C = torch.zeros(incore_loop * num_groups, tile_size, tile_size, dtype=torch.float32)
        return TaskArgsBuilder(
            TensorArg("A", A.flatten()),
            TensorArg("B", B.flatten()),
            TensorArg("C", C.flatten()),
            Scalar("tile_size", ctypes.c_int64(tile_size)),
            Scalar("grid_k", ctypes.c_int64(grid_k)),
            Scalar("num_groups", ctypes.c_int64(num_groups)),
            Scalar("incore_loop", ctypes.c_int64(incore_loop)),
        )

    def compute_golden(self, args, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = args.A.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        B = args.B.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        C = args.C.reshape(incore_loop * num_groups, tile_size, tile_size)
        C[:] = 0.0
        for group in range(num_groups):
            for k_idx in range(grid_k):
                for i in range(incore_loop):
                    C[group * incore_loop + i] += torch.matmul(A[group, k_idx, i], B[group, k_idx, i])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
