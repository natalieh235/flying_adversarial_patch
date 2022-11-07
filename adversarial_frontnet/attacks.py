import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from util import load_dataset, load_model
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    # calculate random translation, rotation (in radians), and scale factor
    # tx = torch.randint(high=2, size=(1,)).float().requires_grad_()    # TODO: fix translation such that patch is always visible
    # ty = torch.randint(high=2, size=(1,)).float().requires_grad_()
    # rotation = torch.distributions.uniform.Uniform(np.radians(-45), np.radians(45)).sample().requires_grad_()  # PyTorch doesn't offer a radians function yet
    # scale = torch.distributions.uniform.Uniform(0.01, 0.7).sample().requires_grad_()

    tx = torch.tensor([0.]).float().requires_grad_()
    ty = torch.tensor([1.]).float().requires_grad_()
    rotation = torch.tensor(np.radians(35.76629)).requires_grad_()
    scale = torch.tensor(0.63830805).requires_grad_()

    print("Rotation: ", np.degrees(rotation.detach().numpy()))
    print("Scale: ", scale.detach().numpy())
    print("Tx: ", tx.detach().numpy(), "Ty: ", ty.detach().numpy())

    image, pose = dataset.dataset.__getitem__(0)
    patch = (torch.rand(1, 1, 50, 50) * 255.).requires_grad_()
    patch_copy = patch.detach().clone()

    image = image.unsqueeze(0)

    print("Original pose: ", pose, pose.shape)
    prediction = torch.concat(model(image)).squeeze(1)
    print("Predicted pose: ", prediction)
    print("L2 dist original-predicted: ", torch.dist(pose, prediction, p=2))
    
    # place the patch
    new_image = place_patch(image, patch, angle=rotation, scale=scale, tx=tx, ty=ty)
    plt.imshow(new_image[0][0].detach().numpy())
    plt.show()

    # initialize optimizer
    opt = torch.optim.Adam([patch], lr=1e-4)


    pred_attack = torch.concat(model(new_image)).squeeze(1)
    print("Predicted pose after attack: ", pred_attack)

    print("L2 dist predicted-attack: ", torch.dist(prediction, pred_attack, p=2))

    target = torch.tensor([0.0, -0.7274,  0.3108, -0.1638])
    print("Target: ", target, target.shape )

    loss = torch.dist(prediction, pred_attack, p=2)
    i = 0.
    while loss > 0.3:
        i += 1
        prediction = torch.concat(model(new_image)).squeeze(1)
        loss = torch.dist(prediction, target, p=2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # patch = torch.clamp(patch, 0., 255.)
        patch.data.clamp_(0., 255.)
        new_image = place_patch(image, patch, angle=rotation, scale=scale, tx=tx, ty=ty)
        if i % 100 == 0:
            print("step %d, loss %.6f" % (i, loss))

    print("Bing!")
    print("Last loss: ", loss.detach().numpy())
    print("Last prediciton: ", prediction)
    print("L2 distance between first patch and last patch: ", torch.dist(patch_copy, patch, p=2))

    plt.imshow(new_image[0][0].detach().numpy())
    plt.show()