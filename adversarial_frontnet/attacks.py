import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

class Patch(torch.nn.Module):
    def __init__(self, device, patch_size=[1, 50, 50]):
        super(Patch, self).__init__()
        patch = (torch.rand(1, *patch_size, device=device) * 255.)#.requires_grad_()
        self.patch = torch.nn.Parameter(patch)

        transformation_min = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]], device=device)
        transformation_max = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]], device=device)

        self.transformation_min = torch.nn.Parameter(transformation_min)
        self.transformation_max = torch.nn.Parameter(transformation_max)

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

        self.x_target = torch.tensor([True, False, False, False]).to(self.device)
        self.y_target = torch.tensor([False, True, False, False]).to(self.device)
        self.z_target = torch.tensor([False, False, True, False]).to(self.device)

        # init random patch
        self.x_patch = Patch(self.device)
        self.y_patch = Patch(self.device)
        self.z_patch = Patch(self.device)

        self.optimizer = torch.optim.Adam([*self.x_patch.parameters(), *self.y_patch.parameters(), *self.z_patch.parameters()], lr = learning_rate)

    def optimize(self, epochs=100000):
        try:
            losses = []
            for i in range(epochs):
                for _ in range(len(self.dataset)):
                    image_batch, pose_batch = next(iter(self.dataset))
                    image_batch = image_batch.to(self.device)
                    pose_batch = pose_batch.to(self.device)

                    # place patch with current parameters in input image
                    input_images_x_min, input_images_x_max = self.x_patch.batch_place(image_batch)
                    input_images_y_min, input_images_y_max = self.y_patch.batch_place(image_batch)
                    input_images_z_min, input_images_z_max = self.z_patch.batch_place(image_batch)

                    target_x_min = pose_batch * ~self.x_target + (1*self.x_target.int())
                    target_x_max = pose_batch * self.x_target + (3.6*self.x_target.int())

                    target_y_min = pose_batch * ~self.y_target + (-2*self.y_target.int())
                    target_y_max = pose_batch * self.y_target + (2*self.y_target.int())

                    target_z_min = pose_batch * ~self.z_target + (-0.5*self.z_target.int())
                    target_z_max = pose_batch * self.z_target + (0.5*self.z_target.int())


                    # get prediction of current pose from NN
                    pred_x_attack_min = torch.stack(self.model(input_images_x_min))
                    pred_x_attack_min = pred_x_attack_min.view(dataset.batch_size, -1)  # get prediction into appropriate shape
                    
                    pred_x_attack_max = torch.stack(self.model(input_images_x_max))
                    pred_x_attack_max = pred_x_attack_max.view(dataset.batch_size, -1)  # get prediction into appropriate shape

                    # calculate loss between target pose and predicted pose
                    loss_min = torch.dist(target_x_min, pred_x_attack_min, p=2)
                    loss_max = torch.dist(target_x_max, pred_x_attack_max, p=2)

                    # average loss between the two loss terms
                    loss_x = (loss_min + loss_max) / 2.

                    pred_y_attack_min = torch.stack(self.model(input_images_y_min))
                    pred_y_attack_min = pred_y_attack_min.view(dataset.batch_size, -1)  # get prediction into appropriate shape
                    
                    pred_y_attack_max = torch.stack(self.model(input_images_y_max))
                    pred_y_attack_max = pred_y_attack_max.view(dataset.batch_size, -1)  # get prediction into appropriate shape

                    # calculate loss between target pose and predicted pose
                    loss_min = torch.dist(target_y_min, pred_y_attack_min, p=2)
                    loss_max = torch.dist(target_y_max, pred_y_attack_max, p=2)

                    # average loss between the two loss terms
                    loss_y = (loss_min + loss_max) / 2.

                    pred_z_attack_min = torch.stack(self.model(input_images_z_min))
                    pred_z_attack_min = pred_z_attack_min.view(dataset.batch_size, -1)  # get prediction into appropriate shape
                    
                    pred_z_attack_max = torch.stack(self.model(input_images_z_max))
                    pred_z_attack_max = pred_z_attack_max.view(dataset.batch_size, -1)  # get prediction into appropriate shape

                    # calculate loss between target pose and predicted pose
                    loss_min = torch.dist(target_z_min, pred_z_attack_min, p=2)
                    loss_max = torch.dist(target_z_max, pred_z_attack_max, p=2)

                    # average loss between the two loss terms
                    loss_z = (loss_min + loss_max) / 2.


                    # loss = torch.stack((loss_x, loss_y, loss_z))
                    # loss = torch.min(loss)                           # this loss will only update one patch!
                    loss = (loss_x + loss_y + loss_z) / 3.

                    # save loss for evaluation
                    losses.append(loss.detach().cpu().numpy())

                    # perform update step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # restrict patch pixel values to stay between 0 and 255
                    self.x_patch.patch.data.clamp_(0., 255.)
                    self.y_patch.patch.data.clamp_(0., 255.)
                    self.z_patch.patch.data.clamp_(0., 255.)

                # occasional print for observering training process
                if i % 1 == 0:
                    print("step %d, loss %.6f"  % (i, loss))
                    print("Patch x")
                    print("Min: ", self.x_patch.transformation_min.detach().cpu().numpy())
                    print("Max: ", self.x_patch.transformation_max.detach().cpu().numpy())
                    print("Patch y")
                    print("Min: ", self.y_patch.transformation_min.detach().cpu().numpy())
                    print("Max: ", self.y_patch.transformation_max.detach().cpu().numpy())
                    print("Patch z")
                    print("Min: ", self.z_patch.transformation_min.detach().cpu().numpy())
                    print("Max: ", self.z_patch.transformation_max.detach().cpu().numpy())
        except KeyboardInterrupt:                   # training process can be interrupted anytime
            print("Aborting optimization...")    

        print("Bing!")
        print("Last loss: ", loss.detach().cpu().numpy())
        print("Last prediciton x: ", pred_x_attack_min, pred_x_attack_max)
        print("Last prediciton y: ", pred_y_attack_min, pred_y_attack_max)
        # print("Last prediciton z: ", pred_z_attack_min, pred_z_attack_max)

        np.save(self.path+'losses_test', losses)

        return self.x_patch


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
    print("Last loss: ", loss.detach().numpy())
    print("Last prediciton: ", pred_attack)

    np.save(path+'losses_test', losses)

    return patch, transformation_matrix #[angle, scale, tx, ty]

if __name__=="__main__":
    # import matplotlib.pyplot as plt
    import os

    from util import load_dataset, load_model
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    # dataset.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # dataset.dataset.labels.to(device)

    path = 'eval/targeted_test/'
    os.makedirs(path, exist_ok = True)

    image, pose = dataset.dataset.__getitem__(0)
    image = image.unsqueeze(0).to(device)
    pose = pose.to(device)
    
    attack = TargetedAttack(model, dataset, device, path=path)
    
    np.save(path+'ori_patch', attack.x_patch.patch.detach().cpu().numpy())

    print("Original pose: ", pose, pose.shape)
    prediction = torch.concat(model(image)).squeeze(1)
    print("Predicted pose: ", prediction)
    print("L2 dist original-predicted: ", torch.dist(pose, prediction, p=2))

    #optimized_x_patch, optimized_transformation = targeted_attack(image, torch.tensor([4., *prediction[1:]]), model, path)
    optimized_x_patch = attack.optimize()
    
    np.save(path+"opti_patch", optimized_x_patch.patch.detach().cpu().numpy())
    new_image_min = place_patch(image, optimized_x_patch.patch, optimized_x_patch.transformation_min)

    # plt.imshow(new_image_min[0][0].detach().cpu().numpy(), cmap='gray')
    # plt.show()

    # new_image_max = place_patch(image, optimized_x_patch.patch, optimized_x_patch.transformation_max)

    # plt.imshow(new_image_max[0][0].detach().cpu().numpy(), cmap='gray')
    # plt.show()