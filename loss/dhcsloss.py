import torch


def dhcsloss(outputs, U, S):
    theta = outputs @ U / 2
    theta = torch.clamp(theta, min=-100, max=50)
    loss = (torch.log(1 + torch.exp(theta)) - S * theta).sum()
    loss = loss / S.numel()
    return loss
