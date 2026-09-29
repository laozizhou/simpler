# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# topo_queue's host-side cases.
#
# These are declared with simpler_ut_add_target rather than through a
# `<runtime>_case` macro, and deliberately so: those macros resolve an include
# contract through simpler_runtime_from_tag(), which only knows runtimes that
# carry a build_config.py. topo_queue's algorithm layer is header-only and
# platform-free, so it needs no contract -- one include directory covers it, and
# the cases build before the runtime is a runtime.
#
# When topo_queue does gain a build_config.py, these move to the same shape as
# the neighbouring directories (a topo_case macro over simpler_ut_runtime_case)
# and gain the a2a3 sibling and tests/ut/cpp/common/topo_queue/ that
# cmake/runtimes.cmake then requires.

set(TOPO_QUEUE_RUNTIME_DIR ${SIMPLER_SRC}/a5/runtime/topo_queue/runtime)

macro(topo_queue_case src)
    simpler_ut_add_target(NAME test_topo_queue_${ARGV1}
        SOURCES ${src}
        INCLUDES ${TOPO_QUEUE_RUNTIME_DIR}
        LABEL no_hardware
        PRIVATE_WORKDIR)
endmacro()

topo_queue_case(test_topo_sort.cpp sort)
topo_queue_case(test_topo_worker.cpp worker)
topo_queue_case(test_platform_contract.cpp platform_contract)
topo_queue_case(test_topo_image.cpp image)
topo_queue_case(test_topo_worker_stress.cpp worker_stress)
topo_queue_case(test_topo_worker_failures.cpp worker_failures)
topo_queue_case(test_topo_type_gate.cpp type_gate)

# The prepare test drives the AICPU's image builder against a byte-level fake of
# HBG's graph, so unlike the algorithm cases it needs HBG's include contract
# (scheduler_graph.h, dispatch_payload.h) and the platform's core_type.h.
simpler_ut_add_target(NAME test_topo_queue_prepare
    SOURCES test_topo_prepare.cpp
    INCLUDES ${TOPO_QUEUE_RUNTIME_DIR}
             ${SIMPLER_SRC}/a5/runtime/host_build_graph/runtime
             ${SIMPLER_SRC}/a5/runtime/host_build_graph/common
             ${SIMPLER_SRC}/a5/runtime
             ${SIMPLER_SRC}/common/host_build_graph
             ${SIMPLER_SRC}/common/platform/include
             ${SIMPLER_SRC}/common/task_interface
             ${SIMPLER_SRC}/common
    LABEL no_hardware
    PRIVATE_WORKDIR)
