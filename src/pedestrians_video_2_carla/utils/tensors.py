import torch


def eye_batch(batch_size, joints, device=torch.device('cpu'), dtype=torch.float32, size=3):
    """
    Returns a batch of identity matrices for the given batch size and number of joints.
    """
    return torch.eye(size, device=device, dtype=dtype).reshape(
        (1, 1, size, size)).repeat((batch_size, joints, 1, 1))
