import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch


class TargetedAttack():
    def __init__(self, model, image, pose, target=[True, False, False, False], learning_rate = 1e-3, path='eval/targeted/'):
        self.model = model
        self.image = image
        self.pose = pose
        
        self.path = path

        # define the maximum and minimum targets to optimize towards
        self.target_min = self.pose * ~target
        self.target_max = self.pose * ~target + (3*target.int())   # TODO: check if this makes sense for y and z

        # init random patch
        self.patch = (torch.rand(1, 1, 50, 50) * 255.).requires_grad_()

        # init random distribution samplers for initial random pose of the patch
        # patch is placed using an angle, a scale factor and a translation vector
        self.angle_sampler = torch.distributions.uniform.Uniform(np.radians(-45), np.radians(45))
        self.scale_sampler = torch.distributions.uniform.Uniform(0.01, 0.5)
        self.translation_sampler = torch.distributions.uniform.Uniform(0.0, 1.0)

        # generate two lists of patch placement parameters
        self.patch_parameters_min = torch.tensor([self.angle_sampler.sample(), self.scale_sampler.sample(), *self.translation_sampler.sample_n(2)]).requires_grad_(True)
        self.patch_parameters_max = torch.tensor([self.angle_sampler.sample(), self.scale_sampler.sample(), *self.translation_sampler.sample_n(2)]).requires_grad_(True)

        self.optimizer = torch.optim.Adam([self.patch, self.patch_parameters_min, self.patch_parameters_max], lr = learning_rate)

    def optimize(self, steps=100000):
        try:
            losses = []
            for i in steps:
                # place patch with current parameters in input image
                input_image_min = place_patch(self.image, self.patch, *self.patch_parameters_min)
                input_image_max = place_patch(self.image, self.patch, *self.patch_parameters_max)

                # get prediction of current pose from NN
                pred_attack_min = torch.concat(self.model(input_image_min)).squeeze(1)
                pred_attack_max = torch.concat(self.model(input_image_max)).squeeze(1)

                # calculate loss between target pose and predicted pose
                loss_min = torch.dist(self.target_min, pred_attack_min, p=2)
                loss_max = torch.dist(self.target_max, pred_attack_max, p=2)

                # average loss between the two loss terms
                loss = (loss_min + loss_max) / 2.

                # save loss for evaluation
                losses.append(loss.detach().numpy())

                # perform update step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # restrict patch pixel values to stay between 0 and 255
                self.patch.data.clamp_(0., 255.)

                # occasional print for observering training process
                if i % 100 == 0:
                    print("step %d, loss %.6f"  % (i, loss))

                    print("Min: ", self.patch_parameters_min.detach().numpy())
                    print("Max: ", self.patch_parameters_max.detach().numpy())
        
        except KeyboardInterrupt:                   # training process can be interrupted anytime
            print("Aborting optimization...")    

        print("Bing!")
        print("Last loss: ", loss.detach().numpy())
        print("Last prediciton: ", pred_attack_min, pred_attack_max)

        np.save(self.path+'losses_test', losses)

        return self.patch, self.patch_parameters_min, self.patch_parameters_max


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

    image, pose = dataset.dataset.__getitem__(0)
    attack = TargetedAttack(model, image.unsqueeze(0), pose, path=path)

    np.save(path+'ori_patch', attack.patch.numpy())

    print("Original pose: ", pose, pose.shape)
    prediction = torch.concat(model(image)).squeeze(1)
    print("Predicted pose: ", prediction)
    print("L2 dist original-predicted: ", torch.dist(pose, prediction, p=2))

    optimized_patch, optimized_pose_right, optimized_pose_left = attack.optimize()
    
    np.save(path+"opti_patch", optimized_patch.detach().numpy())
    new_image_right = place_patch(image, optimized_patch, *optimized_pose_right)

    plt.imshow(new_image_right[0][0].detach().numpy(), cmap='gray')
    plt.show()

    new_image_left = place_patch(image, optimized_patch, *optimized_pose_left)

    plt.imshow(new_image_left[0][0].detach().numpy(), cmap='gray')
    plt.show()
