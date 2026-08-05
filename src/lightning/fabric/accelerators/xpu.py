# Copyright The Lightning AI team.
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
from functools import lru_cache
from typing import Any, Dict, List, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator


class XPUAccelerator(Accelerator):
    """Support for a Intel Discrete Graphics Cards 'XPU'."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    @override
    def parse_devices(devices: Any) -> Any:
        # Put parsing logic here how devices can be passed into the Trainer
        # via the `devices` argument
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids

        return _parse_gpu_ids(devices, include_xpu=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects

        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return num_xpu_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.xpu.is_available()

    @override
    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return {}

    @override
    def setup_device(self, device: torch.device) -> None:
        pass

    @override
    def teardown(self) -> None:
        pass

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description=cls.__name__,
        )


@lru_cache(1)
def num_xpu_devices() -> int:
    """Returns the number of available XPU devices.

    Unlike :func:`torch.xpu.device_count`, this function does its best not to create a XPU context for fork support,
    if the platform allows it.

    """
    return torch.xpu.device_count()


def _get_all_visible_xpu_devices() -> List[int]:
    """Returns a list of all visible Intel XPU devices.

    Devices masked by the environment variabale ``ZE_AFFINITY_MASK`` won't be returned here. For example, assume you
    have 8 physical GPUs. If ``ZE_AFFINITY_MASK="1,3,6"``, then this function will return the list ``[0, 1, 2]``
    because these are the three visible GPUs after applying the mask ``ZE_AFFINITY_MASK``.

    """
    return list(range(num_xpu_devices()))
