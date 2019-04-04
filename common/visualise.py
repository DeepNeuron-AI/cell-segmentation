import torch


def unnormalise(tensor: torch.Tensor):
    """ Converts normalised CxHxW tensor to HxWxC numpy image. """
    tensor = tensor.cpu().detach()
    min, max = float(tensor.min()), float(tensor.max())
    tensor = tensor.clamp_(min=min, max=max)
    tensor = tensor.add_(-min).div_(max - min + 1e-5)
    image = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    if image.shape[-1] == 1:
        image = image.squeeze()
    return image
