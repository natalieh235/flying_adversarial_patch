from email.mime import base
import torch
import numpy as np
import matplotlib.pyplot as plt

# base_grid = torch.empty(3, 50, 50)

# row_0 = torch.linspace(-1, 1, 50)
# row_0 = row_0 * (50. - 1.) / 50.

# row_1 = torch.linspace(-1, 1, 50)
# row_1 = row_1 * (50. - 1.) / 50.

# base_grid[0].copy_(row_0)
# base_grid[1].copy_(row_1)
# base_grid[2].fill_(1)

# grid = base_grid.view(1, 50*50, 3)


# theta = torch.tensor(data=[[113.42067089551912,  0.0, 60.969746425421235],
#                           [0.0, 217.87577756674708, 105.09047406671733], 
#                           [0.0, 0.0, 1.0]]).unsqueeze(0)
# theta_inv = torch.inverse(theta)

# #print(theta_inv, theta_inv.shape)
# #theta_transposed = theta_inv.transpose(1,2)
# #print(theta_transposed, theta_transposed.shape)

# #affine_grid = grid.bmm(theta_transposed).view(1, 50, 50, 2)
# #print(affine_grid, affine_grid.shape)

# ori_affine_grid = torch.nn.functional.affine_grid(theta_inv, (1, 1, 1, 50, 50), align_corners=False)
# print(ori_affine_grid, ori_affine_grid.shape)

h = 10
w = 5

ph = 1
pw = 1

image = torch.zeros(1, 1, h, w).requires_grad_(True)
patch = torch.ones(1, 1, ph, pw).requires_grad_(True)

# h_from = int((h - ph) / 2)
# h_til = int((h + ph) / 2)
# w_from = int((w - pw) / 2)
# w_til = int((w + pw) / 2)

# image[:, :, h_from:h_til, w_from:w_til] = patch
# plt.imshow(image[0][0].numpy())
# plt.show()

### 2d
cos = np.cos(np.radians(30))
sin = np.sin(np.radians(30))

rot_matrix = torch.zeros((1, 3, 3), dtype=torch.float32)
# [ cos, -sin, trans_x]
# [ sin,  cos, trans_y]
# [   0,    0,       1]
# Note that translations should be normalized to [-1.0, 1.0]
rot_matrix[:, 0, 0] = cos
rot_matrix[:, 0, 1] = -sin
rot_matrix[:, 1, 0] = sin
rot_matrix[:, 1, 1] = cos
rot_matrix[:, 0, 2] = 0 # right
rot_matrix[:, 1, 2] = 0 # bottom
rot_matrix[:, 2, 2] = 1

# print(rot_matrix)

inv_rot_matrix = torch.inverse(rot_matrix)[:, :2]

affine_grid = torch.nn.functional.affine_grid(inv_rot_matrix, size=(1, 1, h, w), align_corners=False)

transformed_patch = torch.nn.functional.grid_sample(patch, affine_grid, align_corners=False)
# plt.imshow(transformed_patch[0][0].detach().numpy())
# plt.show()


print("own implementation")
base_grid = torch.empty(3, h, w)
x_grid = torch.linspace(-1, 1, steps=w)
x_grid = x_grid * (w - 1.) / w
base_grid[0].copy_(x_grid)
y_grid = torch.linspace(-1, 1, steps=h).unsqueeze_(-1)
y_grid = y_grid * (h - 1.) / h
base_grid[1].copy_(y_grid)
base_grid[2].fill_(1)
# base_grid[3].fill_(1)
base_grid = base_grid.view(3, -1)
print(base_grid.shape)

u, v, z = rot_matrix.squeeze(0).mT @ base_grid # squeeze because rot matrix is in shape [1, 3, 3]
img_x = u/z
img_y = v/z

grid = torch.stack([img_y, img_x]).mT
grid = grid.reshape((1, h, w, 2))

print("comparison affine_grid and grid")
print("affine_grid: ", affine_grid, affine_grid.shape)
print("grid: ", grid, grid.shape)

transformed_patch_own = torch.nn.functional.grid_sample(patch, grid, align_corners=False)

print("comparison of the final transformed patches")
print("pytorch: ", transformed_patch, transformed_patch.shape)
print("own: ", transformed_patch_own, transformed_patch_own.shape)

print(transformed_patch-transformed_patch_own)
# plt.imshow(transformed_patch_own[0][0].detach().numpy())
# plt.show()
