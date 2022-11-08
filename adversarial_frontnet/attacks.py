import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

def targeted_attack(image, patch, model, angle, scale, tx, ty, path="eval/targeted/"):
    # initialize optimizer
    opt = torch.optim.Adam([patch, angle, scale, tx, ty], lr=1e-3)
    prediction = torch.concat(model(image)).squeeze(1)

    target_left = torch.tensor([0.0, *prediction[1:]])
    target_right = torch.tensor([3.0, *prediction[1:]])

    tx_left = torch.tensor([0.3]).requires_grad_()
    ty_left = torch.tensor([0.0]).requires_grad_()
    angle_left = torch.tensor(np.radians(2.45)).requires_grad_()
    scale_left = torch.tensor(0.26).requires_grad_()
    
    new_image_right = place_patch(image, patch, angle=angle, scale=scale, tx=tx, ty=ty)
    pred_attack_right = torch.concat(model(new_image_right)).squeeze(1)
    loss_right = torch.dist(prediction, pred_attack_right, p=2)

    new_image_left = place_patch(image, patch, angle=angle_left, scale=scale_left, tx=tx_left, ty=ty_left)
    pred_attack_left = torch.concat(model(new_image_left)).squeeze(1)
    loss_left = torch.dist(prediction, pred_attack_left, p=2)   

    loss = loss_left + loss_right

    losses = []
    losses.append(loss.detach().numpy())

    i = 0.
    try:
        while loss > 0.003:
            i += 1
            pred_attack_right = torch.concat(model(new_image_right)).squeeze(1)
            pred_attack_left = torch.concat(model(new_image_left)).squeeze(1)

            loss_right = torch.dist(target_right, pred_attack_right, p=2)
            loss_left = torch.dist(target_left, pred_attack_left, p=2)

            loss = loss_left + loss_right
            # loss = torch.dist(prediction, target, p=2)
            losses.append(loss.detach().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()

            patch.data.clamp_(0., 255.)
            new_image_right = place_patch(image, patch, angle=angle, scale=scale, tx=tx, ty=ty)
            new_image_left = place_patch(image, patch, angle=angle_left, scale=scale_left, tx=tx_left, ty=ty_left)
            if i % 100 == 0:
                print("step %d, loss %.6f"  % (i, loss))
                print("Right: angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (np.degrees(angle.detach().numpy()), scale.detach().numpy(), tx.detach().numpy(), ty.detach().numpy()))
                print("Left: angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (np.degrees(angle_left.detach().numpy()), scale_left.detach().numpy(), tx_left.detach().numpy(), ty_left.detach().numpy()))
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().numpy())
    print("Last prediciton: ", pred_attack_left, pred_attack_right)

    np.save(path+'losses_test', losses)

    return patch, [angle, scale, tx, ty], [angle_left, scale_left, tx_left, ty_left]


def untargeted_attack(image, patch, model, angle, scale, tx, ty, path='eval/untargeted/'):
    # initialize optimizer
    opt = torch.optim.Adam([patch, angle, scale, tx, ty], lr=1e-3)
    prediction = torch.concat(model(image)).squeeze(1)
    
    new_image = place_patch(image, patch, angle=angle, scale=scale, tx=tx, ty=ty)
    # pred_attack = torch.concat(model(new_image)).squeeze(1)
    # loss = -torch.dist(prediction, pred_attack, p=2)

    losses = []
    # losses.append(loss.detach().numpy())

    i = 0.
    try:
        while True:
            i += 1
            pred_attack= torch.concat(model(new_image)).squeeze(1)
            loss = -torch.dist(prediction, pred_attack, p=2)
            losses.append(loss.detach().numpy())
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            # patch.data.clamp_(0., 255.)
            new_image = place_patch(image, patch, angle=angle, scale=scale, tx=tx, ty=ty)
            if i % 100 == 0:
                print("step %d, loss %.6f, angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (i, loss, np.degrees(angle.detach().numpy()), scale.detach().numpy(), tx.detach().numpy(), ty.detach().numpy()))
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().numpy())
    print("Last prediciton: ", pred_attack)

    np.save(path+'losses_test', losses)

    return patch, [angle, scale, tx, ty]

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import os

    from util import load_dataset, load_model
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    path = 'eval/targeted_2_sides/'
    os.makedirs(path, exist_ok = True)

    # calculate random translation, rotation (in radians), and scale factor
    # tx = torch.randint(high=2, size=(1,)).float().requires_grad_()    # TODO: fix translation such that patch is always visible
    # ty = torch.randint(high=2, size=(1,)).float().requires_grad_()
    # rotation = torch.distributions.uniform.Uniform(np.radians(-45), np.radians(45)).sample().requires_grad_()  # PyTorch doesn't offer a radians function yet
    # scale = torch.distributions.uniform.Uniform(0.01, 0.7).sample().requires_grad_()

    tx = torch.tensor([0.2]).float().requires_grad_()
    ty = torch.tensor([0.1]).float().requires_grad_()
    rotation = torch.tensor(np.radians(35.76629)).requires_grad_()
    scale = torch.tensor(0.2).requires_grad_()

    print("Rotation: ", np.degrees(rotation.detach().numpy()))
    print("Scale: ", scale.detach().numpy())
    print("Tx: ", tx.detach().numpy(), "Ty: ", ty.detach().numpy())

    image, pose = dataset.dataset.__getitem__(0)
    patch = (torch.rand(1, 1, 50, 50) * 255.).requires_grad_()
    patch_copy = patch.detach().clone()
    np.save(path+'ori_patch', patch_copy.numpy())

    image = image.unsqueeze(0)

    print("Original pose: ", pose, pose.shape)
    prediction = torch.concat(model(image)).squeeze(1)
    print("Predicted pose: ", prediction)
    print("L2 dist original-predicted: ", torch.dist(pose, prediction, p=2))
    
    # place the patch
    new_image = place_patch(image, patch, angle=rotation, scale=scale, tx=tx, ty=ty)
    plt.imshow(new_image[0][0].detach().numpy(), cmap='gray')
    plt.show()

    pred_attack = torch.concat(model(new_image)).squeeze(1)
    print("Predicted pose after attack: ", pred_attack)

    print("L2 dist predicted-attack: ", torch.dist(prediction, pred_attack, p=2))

    # target = torch.tensor([0.0, -0.7274,  0.3108, -0.1638])
    # print("Target: ", target, target.shape )

    optimized_patch, optimized_pose_right, optimized_pose_left = targeted_attack(image, patch, model, rotation, scale, tx, ty)
    np.save(path+"opti_patch", optimized_patch.detach().numpy())
    new_image_right = place_patch(image, patch, *optimized_pose_right)

    plt.imshow(new_image_right[0][0].detach().numpy(), cmap='gray')
    plt.show()

    new_image_left = place_patch(image, patch, *optimized_pose_left)

    plt.imshow(new_image_left[0][0].detach().numpy(), cmap='gray')
    plt.show()
