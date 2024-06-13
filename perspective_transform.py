import torch
import numpy as np

from src.patch_placement import place_patch

patch = torch.ones(1,1,3,3, dtype=torch.float64)
image = torch.zeros(1,1,96,160)

start = np.array([[0., 0.], [0., 2.], [2., 2.], [2., 0.]])  # 5x5 patch in top left corner  
# end = np.array([[0., 0.], [0., 2.], [2., 2.], [2., 0.]])

def _get_perspective_coeffs(startpoints, endpoints):
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError(
            f"Please provide exactly four corners, got {len(startpoints)} startpoints and {len(endpoints)} endpoints."
        )
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float64)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    # print(a_matrix)

    b_matrix = torch.tensor(startpoints, dtype=torch.float64).view(8)
    # do least squares in double precision to prevent numerical issues
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gelss").solution.to(torch.float32)

    # output: List[float] = res.tolist()
    return res.tolist()

def _perspective_grid(coeffs, ow: int, oh: int):
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]]
    )
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]])

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3)
    x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh])
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)


def _get_inverse_affine_matrix(
    center, angle: float, translate, scale: float, shear, inverted: bool = True
):
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = np.radians(angle)
    sx = np.radians(shear[0])
    sy = np.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = np.cos(rot - sy) / np.cos(sy)
    b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
    c = np.sin(rot - sy) / np.cos(sy)
    d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix




from torchvision.transforms import RandomPerspective
from torchvision.transforms.functional import affine

matrix = torch.tensor(_get_inverse_affine_matrix([96./2., 160./2.], angle=0., translate=[0., 0.], scale=5., shear=[0., 0.]))
print(matrix.shape)
matrix = matrix.reshape(1,2,3)
print(matrix.shape)

affine_grid = torch.affine_grid_generator(matrix, (1, 1, 96, 160), align_corners=False)
# print(affine_grid.shape, affine_grid.dtype)

# _, end = RandomPerspective(distortion_scale=0.2, p=1, interpolation=0).get_params(160, 96, distortion_scale=0.9)
# print(start, end)

# coeffs = _get_perspective_coeffs(start, end)

# coeffs[0] = 0.005
# coeffs[4] = -0.006
# print(np.round(coeffs, decimals=2))

# M = np.zeros(9,)
# M[:-1] = coeffs
# M[-1] = 1.
# M = M.reshape(3,3)
# print(M)

# perspective_grid = _perspective_grid(coeffs, 160, 96)
print(patch.dtype, affine_grid.dtype)
mod_img = torch.nn.functional.grid_sample(patch, affine_grid, mode='nearest', padding_mode='zeros', align_corners=False)

# M[0, 0, 0] = 0.5
# M[0, 1, 1] = 0.5
# print(M[:, :2])
# mod_img = place_patch(image, patch, M[:, :2], random_perspection=False)

import matplotlib.pyplot as plt

plt.imshow(mod_img.numpy()[0][0], cmap='gray')
plt.savefig('example.png')