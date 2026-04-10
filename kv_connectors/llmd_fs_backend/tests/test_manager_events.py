# Copyright 2025 The llm-d Authors.
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

from __future__ import annotations

import importlib
import logging
import sys
import types
from dataclasses import dataclass


def _load_manager_module(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = logging.getLogger

    vllm_v1 = types.ModuleType("vllm.v1")
    vllm_v1_core = types.ModuleType("vllm.v1.core")
    vllm_v1_core_kv = types.ModuleType("vllm.v1.core.kv_cache_utils")
    vllm_v1_core_kv.BlockHash = bytes

    vllm_v1_offload = types.ModuleType("vllm.v1.kv_offload")
    vllm_v1_offload_abstract = types.ModuleType("vllm.v1.kv_offload.abstract")

    class LoadStoreSpec:
        pass

    class OffloadingManager:
        pass

    @dataclass
    class OffloadingEvent:
        block_hashes: list[bytes]
        block_size: int
        medium: str
        removed: bool

    @dataclass
    class PrepareStoreOutput:
        block_hashes_to_store: list[bytes]
        store_spec: object
        block_hashes_evicted: list[bytes]

    vllm_v1_offload_abstract.LoadStoreSpec = LoadStoreSpec
    vllm_v1_offload_abstract.OffloadingManager = OffloadingManager
    vllm_v1_offload_abstract.OffloadingEvent = OffloadingEvent
    vllm_v1_offload_abstract.PrepareStoreOutput = PrepareStoreOutput

    modules = {
        "vllm": vllm,
        "vllm.logger": vllm_logger,
        "vllm.v1": vllm_v1,
        "vllm.v1.core": vllm_v1_core,
        "vllm.v1.core.kv_cache_utils": vllm_v1_core_kv,
        "vllm.v1.kv_offload": vllm_v1_offload,
        "vllm.v1.kv_offload.abstract": vllm_v1_offload_abstract,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    for module_name in ["llmd_fs_backend.manager", "llmd_fs_backend.mediums"]:
        sys.modules.pop(module_name, None)

    return importlib.import_module("llmd_fs_backend.manager")


def test_complete_store_no_events_when_disabled(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    manager = manager_module.SharedStorageOffloadingManager(
        file_mapper=object(),
        block_size=256,
        enable_events=False,
    )

    manager.complete_store([b"a", b"b"], success=True)

    assert list(manager.take_events()) == []


def test_complete_store_records_event_when_enabled(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    manager = manager_module.SharedStorageOffloadingManager(
        file_mapper=object(),
        block_size=512,
        enable_events=True,
    )

    hashes = [b"a", b"b"]
    manager.complete_store(hashes, success=True)

    events = list(manager.take_events())
    assert len(events) == 1
    assert events[0].block_hashes == hashes
    assert events[0].block_size == 512
    assert events[0].medium == "SHARED_STORAGE"
    assert events[0].removed is False


def test_complete_store_does_not_record_failed_store(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    manager = manager_module.SharedStorageOffloadingManager(
        file_mapper=object(),
        block_size=256,
        enable_events=True,
    )

    manager.complete_store([b"a"], success=False)

    assert list(manager.take_events()) == []


def test_take_events_clears_buffer(monkeypatch):
    manager_module = _load_manager_module(monkeypatch)
    manager = manager_module.SharedStorageOffloadingManager(
        file_mapper=object(),
        block_size=256,
        enable_events=True,
    )

    manager.complete_store([b"a"], success=True)

    first = list(manager.take_events())
    second = list(manager.take_events())

    assert len(first) == 1
    assert second == []
