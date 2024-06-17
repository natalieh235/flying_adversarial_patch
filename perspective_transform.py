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
    # source: https://github.com/pytorch/pytorch/issues/100526#issuecomment-1610226058
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    # theta1 = torch.tensor(
    #     [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]], dtype=dtype, device=device
    # )
    batch_size = coeffs.shape[0]
    theta1 = coeffs[..., :6].reshape(batch_size, 2, 3)

    theta2 = coeffs[..., 6:].repeat_interleave(2, dim=0).reshape(batch_size, 2, 3)

    d = 0.5
    base_grid = torch.empty(batch_size, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    shape = (batch_size, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))

    if center is not None:
        center = torch.tensor(center, dtype=dtype, device=device)
    else:
        center = 1.0

    output_grid = output_grid1.div_(output_grid2).sub_(center)
    return output_grid.view(batch_size, oh, ow, 2)

patch_height = 30
patch_width = 30

patch = torch.ones(2, 1,patch_height,patch_width)

height = 96
width= 160


# start = [[0.0, 0.0], [4.0, 0.], [4., 4.], [0., 4.]]
# end = [[0.0, 0.0], [80.0, 0.], [80., 80.], [0., 80.]]

# coeffs = _get_perspective_coeffs(start, end)

# print(np.round(coeffs, decimals=2))

# output matrix:
# [[0.05, 0.  , 0.  ],
# [0.  , 0.05, 0.  ],
# [0.  , 0.  , 1.  ]])
# which is the inverse of:
# [[20.,  0.,  0.],
# [ 0., 20.,  0.],
# [ 0.,  0.,  1.]]
# --> opencv output

sf = torch.tensor(2.).requires_grad_(True)
tx = torch.tensor(30.).requires_grad_(True)
ty = torch.tensor(60.).requires_grad_(True)

M = torch.eye(3,3)
M[:2, :2] *= sf
M[0, 2] = tx
M[1, 2] = ty
print(M)

M_inv = torch.inverse(M)
coeffs = M_inv.flatten()

sf = torch.tensor(1.).requires_grad_(True)
tx = torch.tensor(80.).requires_grad_(True)
ty = torch.tensor(10.).requires_grad_(True)

M = torch.eye(3,3)
M[:2, :2] *= sf
M[0, 2] = tx
M[1, 2] = ty
#M[2, :2] = torch.tensor([0.5, 0.6]) 
print(M)

M_inv = torch.inverse(M)

coeffs = torch.vstack([coeffs, M_inv.flatten()])
print(coeffs, coeffs.shape)

grid = _perspective_grid(
    coeffs, w=patch_width, h=patch_height, ow=width, oh=height, 
    dtype=torch.float32, device="cpu",
    center = [1., 1.]
)

print(grid.shape)

#output = _apply_grid_transform(patch, grid, "nearest", 0)
output = torch.nn.functional.grid_sample(patch, grid, "nearest", 'zeros', align_corners=True)
print(output.shape)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].imshow(output[0][0].detach().numpy(), cmap='gray')
axs[1].imshow(output[1][0].detach().numpy(), cmap='gray')
plt.savefig('example.png')