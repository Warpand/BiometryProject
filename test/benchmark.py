import gc
import timeit

import torch

from modules.resnet import get_resnet_builder

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    command = "torch.nn.functional.mse_loss(resnet(x), y).backward()"
    x = torch.randn(size=(512, 3, 224, 224)).to("cuda")
    y = torch.randn(size=(512, 512)).to("cuda")

    resnet = get_resnet_builder(512).build(False)
    resnet = resnet.to("cuda")
    resnet(x)
    print(timeit.timeit(command, number=10, globals=globals()))

    gc.collect()
    torch.cuda.empty_cache()

    resnet_compiled = torch.compile(resnet)
    resnet_compiled(x)
    print(timeit.timeit(command, number=10, globals=globals()))
