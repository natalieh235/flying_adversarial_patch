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

h = 96
w = 160

ph = 50
pw = 50

image = torch.zeros(1, 1, h, w).requires_grad_(True)
patch = torch.ones(1, 1, ph, pw).requires_grad_(True)

# h_from = int((h - ph) / 2)
# h_til = int((h + ph) / 2)
# w_from = int((w - pw) / 2)
# w_til = int((w + pw) / 2)

# image[:, :, h_from:h_til, w_from:w_til] = patch
# plt.imshow(image[0][0].numpy())
# plt.show()

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

inv_rot_matrix = torch.inverse(rot_matrix)[:, :2]

affine_grid = torch.nn.functional.affine_grid(inv_rot_matrix, size=(1, 1, h, w), align_corners=False)

transformed_patch = torch.nn.functional.grid_sample(patch, affine_grid, align_corners=False)
plt.imshow(transformed_patch[0][0].detach().numpy())
plt.show()

rot_matrix = torch.zeros((1, 4, 4), dtype=torch.float32)
rot_matrix[:, 0, 0] = cos
rot_matrix[:, 0, 1] = -sin
rot_matrix[:, 1, 0] = sin
rot_matrix[:, 1, 1] = cos
rot_matrix[:, 0, 2] = 0 # right
rot_matrix[:, 1, 2] = 0 # bottom
rot_matrix[:, 3, 3] = 1


inv_rot_matrix = torch.pinverse(rot_matrix)[:, :3]
print(inv_rot_matrix, inv_rot_matrix.shape)

affine_grid = torch.nn.functional.affine_grid(inv_rot_matrix, size=(1, 1, 1, h, w), align_corners=False)

transformed_patch = torch.nn.functional.grid_sample(patch.unsqueeze(0), affine_grid, align_corners=False)
plt.imshow(transformed_patch[0][0][0].detach().numpy())
plt.show()

# base_grid = torch.empty(3, h, w)

# row_0 = torch.linspace(-1, 1, steps=w)
# #row_0 = (row_0 * (w - 1.) / w)

# row_1 = torch.linspace(-1, 1, steps=h).unsqueeze(-1)
# #row_1 = (row_1 * (h - 1.) / h).unsqueeze_(-1)


# base_grid[0].copy_(row_0)
# base_grid[1].copy_(row_1)
# base_grid[2].fill_(1)

# grid = base_grid.view(1, h*w, 3)


# theta_inv = torch.inverse(rot_matrix)[:, :2]

# print(theta_inv, theta_inv.shape)
# #theta_transposed = theta_inv.transpose(1,2)
# #print(theta_transposed, theta_transposed.shape)

# #affine_grid = grid.bmm(theta_transposed).view(1, h, w, 2)
# u, v, z = (rot_matrix @ grid.view(3, -1)).squeeze(0)
# affine_grid = torch.stack([u/z, v/z]).view(1, h, w, 2)
# print(affine_grid, affine_grid.shape)


# rot_matrix = torch.tensor([[0.36, 0.48, -0.80, 0.5],
#               [-0.8, 0.6, 0.0, 0.5],
#               [0.48, 0.64, 0.60, 0.0],
#               [0.0, 0.0, 0.0, 1.0]])

# inv_rot_matrix = torch.pinverse(rot_matrix)[:, :3]
# print(inv_rot_matrix, inv_rot_matrix.shape)

#affine_grid = torch.nn.functional.affine_grid(inv_rot_matrix.mT.unsqueeze(0), size=(1, 1, 1, 96, 160), align_corners=False)
