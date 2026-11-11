"""Which devices the host can run on - asked of the Paddle build installed here.

The devices are the Paddle build's business, not PaddleOCR's: the CPU build sees no card, a
paddlepaddle-gpu build sees the NVIDIA cards its CUDA version and the driver allow. Choosing the
build is the install's job (Jira GATEWAY-47); this only reports what the installed one sees.
"""

from __future__ import annotations

from typing import Protocol

GIB = 2**30


class PaddleDevices(Protocol):
    def device_count(self) -> int: ...
    def name(self, i: int) -> str: ...
    def capability(self, i: int) -> tuple[int, int]: ...
    def memory(self, i: int) -> int: ...


class _InstalledPaddle:
    def __init__(self):
        import paddle
        self._cuda = paddle.device.cuda

    def device_count(self) -> int:
        return self._cuda.device_count()

    def name(self, i: int) -> str:
        return self._cuda.get_device_name(i)

    def capability(self, i: int) -> tuple[int, int]:
        return self._cuda.get_device_capability(i)

    def memory(self, i: int) -> int:
        return self._cuda.get_device_properties(i).total_memory


class Devices:
    def __init__(self, paddle: PaddleDevices):
        self._devices = [{"id": "cpu"}]
        for i in range(paddle.device_count()):
            major, minor = paddle.capability(i)
            self._devices.append({"id": f"gpu:{i}", "name": paddle.name(i),
                                  "compute": f"{major}.{minor}",
                                  "memory_gb": round(paddle.memory(i) / GIB, 1)})

    @classmethod
    def from_paddle(cls) -> "Devices":
        return cls(_InstalledPaddle())

    def list(self) -> list[dict]:
        return self._devices

    def default(self) -> str:
        """The first card if there is one, else the CPU."""
        return self._devices[1]["id"] if len(self._devices) > 1 else "cpu"
