import torch
import time
import torch.nn as nn

import numpy as np
import random

from torchvision import models
from src.models.ee_cnn_residual import EE_CNN_Residual

device = "cuda"
model = EE_CNN_Residual(
    input_shape=(3, 280, 280),
    num_ee=2,
    init_planes=64,
    planes=[64, 128, 256, 512],
    num_classes=3,
    repetitions=[2, 2, 2, 2],
    exit_threshold=0.3,
).to(device)

# model = models.resnet18(pretrained=True)
model.eval()

batch_test_sizes = [64, 128, 256, 512]
num_tests = 20
for batch_size in batch_test_sizes:
    for_loop_times = list()
    batch_times = list()
    print(f"Batch size: {batch_size}")
    for i in range(num_tests):
        print(f"{i}/{num_tests}")
        images = torch.rand((batch_size, 3, 280, 280)).to(device)

        st_for_loop = time.time()
        with torch.no_grad():
            for idx in range(images.shape[0]):
                pred = model(images[idx, ...].unsqueeze(0))

        et_for_loop = time.time()
        elapsed_time_for_loop = et_for_loop - st_for_loop
        for_loop_times.append(elapsed_time_for_loop)

        st_batch = time.time()
        with torch.no_grad():
            pred = model(images)

        et_batch = time.time()
        elapsed_time_batch = et_batch - st_batch
        batch_times.append(elapsed_time_batch)


    print(f"Mean time for-loop {batch_size}: {np.mean(for_loop_times):5f} seconds")
    print(f"Mean time batch {batch_size}: {np.mean(batch_times):5f} seconds")
