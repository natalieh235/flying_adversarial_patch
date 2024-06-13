import torch
import numpy as np

from torchvision.transforms.v2.functional._geometry import _get_inverse_affine_matrix, _compute_affine_output_size, _apply_grid_transform,  _get_perspective_coeffs

from typing import List


def _perspective_grid(
    coeffs: List[float], 
    w: int, h: int, 
    ow: int, oh: int, 
    dtype: torch.dtype, 
    device: torch.device,
    center = None,
) -> torch.Tensor:
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]], dtype=dtype, device=device
    )
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]], dtype=dtype, device=device)

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    shape = (1, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    if center is not None:
        center = torch.tensor(center, dtype=dtype, device=device)
    else:
        center = 1.0

    output_grid = output_grid1.div_(output_grid2).sub_(center)
    return output_grid.view(1, oh, ow, 2)


patch = torch.ones(5,5)

height = 96
width= 160


start = [[0.0, 0.0], [4.0, 0.], [4., 4.], [0., 4.]]
end = [[0.0, 0.0], [80.0, 0.], [80., 80.], [0., 80.]]

coeffs = _get_perspective_coeffs(start, end)
print(np.round(coeffs, decimals=2))

# output matrix:
# [[0.05, 0.  , 0.  ],
# [0.  , 0.05, 0.  ],
# [0.  , 0.  , 1.  ]])
# which is the inverse of:
# [[20.,  0.,  0.],
# [ 0., 20.,  0.],
# [ 0.,  0.,  1.]]
# --> opencv output


grid = _perspective_grid(
    coeffs, w=5, h=5, ow=width, oh=height, 
    dtype=torch.float32, device="cpu",
    center = [1., 1.]
)

print(grid.shape)

output = _apply_grid_transform(patch[None, ...], grid, "nearest", 0)

import matplotlib.pyplot as plt
plt.imshow(output[0].numpy(), cmap='gray')
plt.savefig('example.png')