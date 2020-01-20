import torch
import numpy as np


def image2tensor(image):
    tensor = torch.FloatTensor(np.array(image)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    return tensor
