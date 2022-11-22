import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

from util import plot_patch

class Patch(torch.nn.Module):
    def __init__(self, device, target=[True, False, False, False], patch_size=[1, 50, 50]):
        super(Patch, self).__init__()
        patch = (torch.rand(1, *patch_size, device=device) * 255.)#.requires_grad_()
        self.patch = torch.nn.Parameter(patch)

        transformation_min = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]], device=device)
        transformation_max = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]], device=device)

        self.transformation_min = torch.nn.Parameter(transformation_min)
        self.transformation_max = torch.nn.Parameter(transformation_max)

        self.target = torch.tensor(target).to(device)

        if self.target[0] == True:
            self.lower_limit = 1. 
            self.upper_limit = 3.6
        elif self.target[1] == True:
            self.lower_limit = -2.
            self.upper_limit = 2.
        elif self.target[2] == True:
            self.lower_limit= -0.5
            self.upper_limit= 0.5
        else:
            raise NotImplementedError


    def place(self, image):
        image_min = place_patch(image, self.patch, self.transformation_min)
        image_max = place_patch(image, self.patch, self.transformation_max)
        return torch.stack([image_min, image_max])

    def batch_place(self, batch):
        # out = torch.stack([self.place(img) for img in batch]).view((2, *batch.shape))
        # batch_min, batch_max = out
        batch_min = place_patch(batch, self.patch, self.transformation_min)
        batch_max = place_patch(batch, self.patch, self.transformation_max)

        return batch_min, batch_max

class TargetedAttack():
    def __init__(self, model, dataset, device, learning_rate = 3e-4, path='eval/targeted/'):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self.path = path

        self.x_target = [True, False, False, False]
        self.y_target = [False, True, False, False]
        self.z_target = [False, False, True, False]

        # init random patch
        self.x_patch = Patch(self.device, self.x_target)
        self.y_patch = Patch(self.device, self.y_target)
        self.z_patch = Patch(self.device, self.z_target)

        self.optimizer = torch.optim.Adam([*self.x_patch.parameters(), *self.y_patch.parameters(), *self.z_patch.parameters()], lr = learning_rate)

    def optimize(self, epochs=100000):
        try:
            self.all_losses = []
            self.all_avg_losses = []
            for i in trange(epochs):
                self.epoch_losses = []
                for _ in range(4):#len(self.dataset)):
                    image_batch, pose_batch = next(iter(self.dataset))
                    image_batch = image_batch.to(self.device)
                    pose_batch = pose_batch.to(self.device)

                    self.loss = self._calc_all_losses(image_batch, pose_batch, patches=[self.x_patch, self.y_patch, self.z_patch])

                    # save loss for evaluation
                    self.epoch_losses.append(self.loss.detach().cpu().numpy())
                    self.all_losses.append(self.loss.detach().cpu().numpy())

                    # perform update step
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()

                    # restrict patch pixel values to stay between 0 and 255
                    self.x_patch.patch.data.clamp_(0., 255.)
                    self.y_patch.patch.data.clamp_(0., 255.)
                    self.z_patch.patch.data.clamp_(0., 255.)

                # generate plots and save intermediate patches
                self.all_avg_losses.append(np.mean(self.epoch_losses))
                
                # only save plots after certain epochs
                if i % (epochs/100.) == 0:
                    self._save_plots(i)

                np.save(self.path+'x_patch', self.x_patch.patch.detach().cpu().numpy(), self.x_patch.transformation_min.detach().cpu().numpy(), self.x_patch.transformation_max.detach().cpu().numpy())
                np.save(self.path+'y_patch', self.y_patch.patch.detach().cpu().numpy(), self.y_patch.transformation_min.detach().cpu().numpy(), self.y_patch.transformation_max.detach().cpu().numpy())
                np.save(self.path+'z_patch', self.z_patch.patch.detach().cpu().numpy(), self.z_patch.transformation_min.detach().cpu().numpy(), self.z_patch.transformation_max.detach().cpu().numpy())

                self._save_loss()

        except KeyboardInterrupt:                   # training process can be interrupted anytime
            print("Aborting optimization...")    

        return self.x_patch, self.y_patch, self.z_patch

    def _calc_targets(self, org_poses, target, lower_limit, upper_limit):
        target_min = org_poses * ~target + (lower_limit*target.int())
        target_max = org_poses * ~target + (upper_limit*target.int())

        return target_min, target_max

    def _calc_single_loss(self, images, target, norm=2):
        # get prediction of current pose from NN
        prediciton = torch.stack(self.model(images)).permute(1, 0, 2).squeeze(2)  # get prediction into appropriate shape
        loss = torch.dist(target, prediciton, p=norm)

        return loss

    def _calc_all_losses(self, images, poses, patches):

        losses = torch.zeros(1, device=self.device)
        for i, patch in enumerate(patches):
            images_min, images_max = patch.batch_place(images)
            target_min, target_max = self._calc_targets(poses, patch.target, lower_limit=patch.lower_limit, upper_limit=patch.upper_limit)
            loss_min = self._calc_single_loss(images_min, target_min)
            loss_max = self._calc_single_loss(images_max, target_max)
            losses += (loss_min + loss_max) / 2.

        return losses / len(patches)

    def _save_plots(self, idx, path='plots/epoch_'):
        image, _ = dataset.dataset.__getitem__(0)
        image = image.unsqueeze(0).to(device)

        plot_path = self.path + path + str(idx) + '/'
        os.makedirs(plot_path, exist_ok = True)

        plot_patch(self.x_patch, image, title='X Patch', save=True, path=plot_path)

        plot_patch(self.y_patch, image, title='Y Patch', save=True, path=plot_path)

        plot_patch(self.z_patch, image, title='Z Patch', save=True, path=plot_path)


    def _save_loss(self, path='losses/'):
        save_path = self.path+path
        os.makedirs(save_path, exist_ok = True)
        np.save(save_path+'all_losses', self.all_losses)
        np.save(save_path+'all_avg_losses', self.all_avg_losses)

def targeted_attack(image, patch, target, model, transformation_matrix, path="eval/targeted/"):
    # initialize optimizer
    opt = torch.optim.Adam([transformation_matrix], lr=1e-2)
    prediction = torch.concat(model(image)).squeeze(1)

    new_image = place_patch(image, patch, transformation_matrix)
    pred_attack = torch.concat(model(new_image)).squeeze(1)
    loss = torch.dist(prediction, pred_attack, p=2)

    losses = []
    losses.append(loss.detach().cpu().numpy())

    i = 0.
    try:
        while loss > 0.01:
            i += 1
            prediction = torch.concat(model(new_image)).squeeze(1)
            loss = torch.dist(prediction[1], target[1], p=2)
            losses.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()

            # patch.data.clamp_(0., 255.)
            new_image = place_patch(image, patch, transformation_matrix)
            
            if i % 100 == 0:
                print("step %d, loss %.6f" % (i, loss.detach().cpu().numpy()))
                print("matrix: ", transformation_matrix.view(-1).detach().cpu().numpy())
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().cpu().numpy())
    print("Last prediciton: ", prediction)

    np.save(path+'losses_test', losses)

    return patch, transformation_matrix

def untargeted_attack(image, patch, model, transformation_matrix, path='eval/untargeted/'): # angle, scale, tx, ty,
    # initialize optimizer
    opt = torch.optim.Adam([transformation_matrix], lr=1e-4)
    prediction = torch.concat(model(image)).squeeze(1)
    
    
    new_image = place_patch(image, patch, transformation_matrix)
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
            new_image = place_patch(image, patch, transformation_matrix)
            if i % 100 == 0:
                print("step %d, loss %.6f" % (i, loss))
                print("transformation matrix: ", transformation_matrix.detach().numpy())
                #print("step %d, loss %.6f, angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (i, loss, np.degrees(angle.detach().numpy()), scale.detach().numpy(), tx.detach().numpy(), ty.detach().numpy()))
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().cpu().numpy())
    print("Last prediciton: ", pred_attack)

    np.save(path+'losses_test', losses)

    return patch, transformation_matrix #[angle, scale, tx, ty]

if __name__=="__main__":
    # import matplotlib.pyplot as plt
    import os

    from util import load_dataset, load_model
    model_path = '/home/hanfeld/adversarial_frontnet/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '/home/hanfeld/adversarial_frontnet/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    # dataset.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # dataset.dataset.labels.to(device)

    path = 'eval/targeted_custom_patch_zero_matrix/'
    os.makedirs(path, exist_ok = True)

    patch = np.load("/home/hanfeld/adversarial_frontnet/misc/custom_patch.npy")
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
    
    #transformation_matrix = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]], device=device).requires_grad_(True)
    transformation_matrix = torch.zeros((1,2,3), device=device)
    transformation_matrix[..., 0, 0] = 1.
    transformation_matrix[..., 1, 1] = 1.
    transformation_matrix = transformation_matrix.requires_grad_(True)

    image, pose = dataset.dataset.__getitem__(0)
    image = image.unsqueeze(0).to(device)
    
    pose[1] = -2.
    target = pose.to(device)

    np.save(path+'ori_matrix', transformation_matrix.detach().cpu().numpy())

    _, optimized_matrix = targeted_attack(image, patch, target, model, transformation_matrix, path)

    np.save(path+'optimized_matrix', optimized_matrix.detach().cpu().numpy())



    # image, pose = dataset.dataset.__getitem__(0)
    # image = image.unsqueeze(0).to(device)
    # pose = pose.to(device)
    
    # attack = TargetedAttack(model, dataset, device, path=path)
    
    # np.save(path+'ori_patch', attack.x_patch.patch.detach().cpu().numpy())

    # print("Original pose: ", pose, pose.shape)
    # prediction = torch.concat(model(image)).squeeze(1)
    # print("Predicted pose: ", prediction)
    # print("L2 dist original-predicted: ", torch.dist(pose, prediction, p=2))

    # #optimized_x_patch, optimized_transformation = targeted_attack(image, torch.tensor([4., *prediction[1:]]), model, path)
    # optimized_x_patch, optimized_y_patch, optimized_z_patch = attack.optimize()
    
    # np.save(path+"opti_patch", optimized_x_patch.patch.detach().cpu().numpy())
    # new_image_min = place_patch(image, optimized_x_patch.patch, optimized_x_patch.transformation_min)