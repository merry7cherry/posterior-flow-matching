"""Native single image crop, adapted from robomimic (see third-party notices)."""
import torch
from torch import nn
from torchvision.transforms.functional import center_crop


class CropRandomizer(nn.Module):
    def __init__(self, input_shape, crop_height, crop_width):
        super().__init__()
        self.input_shape = input_shape
        self.crop_height, self.crop_width = crop_height, crop_width

    def output_shape_in(self, input_shape=None):
        return [self.input_shape[0], self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def forward_in(self, images):
        if not self.training:
            return center_crop(images, (self.crop_height, self.crop_width))
        batch, channels, height, width = images.shape
        ch, cw = self.crop_height, self.crop_width
        # Preserve the upstream crop sampling range and RNG order.
        top = ((height-ch)*torch.rand(batch, 1).to(images.device)).long()
        left = ((width-cw)*torch.rand(batch, 1).to(images.device)).long()
        rows = torch.arange(ch, device=images.device)[None, :, None] + top[:, :, None]
        cols = torch.arange(cw, device=images.device)[None, None, :] + left[:, :, None]
        indices = (rows*width+cols).reshape(batch, 1, ch*cw).expand(-1, channels, -1)
        return images.flatten(2).gather(2, indices).reshape(batch, channels, ch, cw)

    def forward_out(self, inputs):
        return inputs

    def forward(self, inputs):
        return self.forward_in(inputs)
