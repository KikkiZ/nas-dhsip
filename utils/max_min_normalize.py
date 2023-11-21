import torch


def tensor_repeat(inputs, x, y):
    inputs = inputs.repeat(x, y, 1)
    inputs = torch.transpose(inputs, 0, 2)
    inputs = torch.transpose(inputs, 1, 2)
    return inputs


def max_min_normalize(inputs: torch.Tensor):
    max_tensor = torch.max(inputs, dim=1).values
    max_tensor = torch.max(max_tensor, dim=1).values
    max_tensor = tensor_repeat(max_tensor, x=inputs.shape[1], y=inputs.shape[2])

    min_tensor = torch.min(inputs, dim=1).values
    min_tensor = torch.min(min_tensor, dim=1).values
    min_tensor = tensor_repeat(min_tensor, x=inputs.shape[1], y=inputs.shape[2])

    return (inputs - min_tensor) / (max_tensor - min_tensor)
