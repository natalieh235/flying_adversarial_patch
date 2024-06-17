import torch
import numpy as np

from torchvision.transforms.v2.functional._geometry import _get_inverse_affine_matrix, _compute_affine_output_size, _apply_grid_transform,  _get_perspective_coeffs

from typing import List

from src.util import load_model

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





if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()


    target = torch.tensor([[1., 1., 0.]], device=device)
    print("Target: ", target)

    patch_height = 96   
    patch_width = 160

    patch = torch.ones(1, 1,patch_height,patch_width, device=device) * 255.

    height = 96
    width= 160

    base_img = torch.rand(1, 1, height, width, device=device) * 255.
    x, y, z, yaw = model(base_img)
    out_base = torch.hstack([x, y, z, yaw])
    print("Output original random image: ", out_base)




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

    sf = torch.tensor([0.3], device=device).requires_grad_(True)
    tx = torch.tensor([0.], device=device).requires_grad_(True)
    ty = torch.tensor([0.], device=device).requires_grad_(True)

    # positions = torch.rand(1, 2, 3, device=device).requires_grad_(True)

    opt = torch.optim.Adam([sf, tx, ty], lr=0.001)


    for i in range(10000):
        # sf, tx, ty = positions
        # eye = torch.eye(2,2, device=device).unsqueeze(0)
        # scale = eye * sf

        # translation_vector = torch.stack([tx, ty]).unsqueeze(0)
        # transformation_matrix = torch.cat([scale, translation_vector], dim=2)
    
        # # print(transformation_matrix, transformation_matrix.shape)

        # last_row = torch.tensor([[0, 0, 1]], device=transformation_matrix.device)
        # # print(last_row.shape)
        # full_matrix = torch.cat([transformation_matrix, last_row.unsqueeze(0)], dim=1)
        # # print(full_matrix)

        # inv_t_matrix = torch.inverse(full_matrix)[:, :2]

        # M = torch.eye(3,3, device=device)
        # M[:2, :2] *= sf
        # M[0, 2] = tx
        # M[1, 2] = ty
        # print(M)

        # M_inv = torch.inverse(M)
        # coeffs = M_inv.flatten().unsqueeze(0)

        # sf = torch.tensor(1.).requires_grad_(True)
        # tx = torch.tensor(80.).requires_grad_(True)
        # ty = torch.tensor(10.).requires_grad_(True)

        M = torch.eye(3,3).to(device)
        M[:2, :2] *= sf
        M[0, 2] = tx
        M[1, 2] = ty
        # #M[2, :2] = torch.tensor([0.5, 0.6]) 
        # print(M)

        M_inv = torch.inverse(M)
        coeffs = M_inv.flatten().unsqueeze(0)

        # coeffs = torch.vstack([coeffs, M_inv.flatten()])
        # print(coeffs, coeffs.shape)

        grid = _perspective_grid(
            coeffs, w=patch_width, h=patch_height, ow=width, oh=height, 
            dtype=torch.float32, device=device,
            center = [1., 1.]
        )

        # grid = torch.nn.functional.affine_grid(inv_t_matrix, size=(1, 1, height, width), align_corners=True)

        # print(grid.shape) 

        #output = _apply_grid_transform(patch, grid, "nearest", 0)
        mask = torch.ones_like(patch, device=patch.device)
        perspected_patch = torch.nn.functional.grid_sample(patch, grid, "bilinear", 'zeros', align_corners=True)
        bit_mask = torch.nn.functional.grid_sample(mask, grid, "bilinear", 'zeros', align_corners=True)
        
        modified_image = base_img * ~bit_mask.bool()
        # and now replace these values with the transformed patch
        modified_image += perspected_patch

        out_mod = torch.hstack(model(modified_image))
        # print("Output with patch: ", out_mod)

        loss = torch.nn.functional.mse_loss(out_mod[..., :3], target)
        
        if i % 1000 == 0:
            print(f"[{i}] loss: {loss} ")

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("-- sf, tx, ty after opt --")
    print(sf, tx, ty)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, modified_image.shape[0], constrained_layout=True)
    for i in range(modified_image.shape[0]):
        axs.imshow(modified_image[i][0].detach().cpu().numpy(), cmap='gray')
    plt.savefig('example.png')