"""Does Paddle in this environment compute on the GPU? Exit 0 only if it does.

The CPU build answers is_compiled_with_cuda() with False and the run_check() of a CUDA build that
cannot reach the card fails, so either is a failed install, not a slow one.
"""

import sys

import paddle

print("paddle", paddle.__version__)
if not paddle.device.is_compiled_with_cuda():
    print("FAIL: this Paddle is a CPU build")
    sys.exit(1)
print("CUDA", paddle.version.cuda(), "cuDNN", paddle.version.cudnn())
print("devices", paddle.device.cuda.device_count(), paddle.device.cuda.get_device_name(0))
paddle.utils.run_check()
print("OK: Paddle computes on the GPU")
