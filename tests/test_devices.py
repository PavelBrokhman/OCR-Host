"""Which devices pavlogix-paddle can run on - asked of the Paddle build installed here."""

from pavlogix_paddle.devices import Devices


class FakePaddle:
    """The part of Paddle's device API Devices asks."""

    def __init__(self, cards):
        self._cards = cards     # [(name, (major, minor), bytes)]

    def device_count(self):
        return len(self._cards)

    def name(self, i):
        return self._cards[i][0]

    def capability(self, i):
        return self._cards[i][1]

    def memory(self, i):
        return self._cards[i][2]


CARD = ("NVIDIA GeForce RTX 2070 SUPER", (7, 5), 8 * 2**30)


def test_the_cpu_is_always_there():
    assert Devices(FakePaddle([])).list()[0] == {"id": "cpu"}


def test_every_card_is_listed_with_its_name_capability_and_memory():
    devices = Devices(FakePaddle([CARD])).list()
    assert devices[1] == {"id": "gpu:0", "name": "NVIDIA GeForce RTX 2070 SUPER",
                          "compute": "7.5", "memory_gb": 8.0}


def test_the_first_card_is_the_default():
    assert Devices(FakePaddle([CARD, CARD])).default() == "gpu:0"


def test_without_a_card_the_default_is_the_cpu():
    assert Devices(FakePaddle([])).default() == "cpu"


def test_the_installed_paddle_answers():
    # Real Paddle: whatever the machine has, the CPU is listed and a default is chosen.
    devices = Devices.from_paddle()
    assert devices.list()[0] == {"id": "cpu"}
    assert devices.default() in [d["id"] for d in devices.list()]
